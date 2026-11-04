"""Lazy torch scorer. Equal-source sampling, group-disjoint selection, full refit."""

import json
from pathlib import Path

import numpy as np

from ..artifacts import freeze, save
from ..config.references import values as representation_values
from ..core import build, register


class EpisodeBank:
    """Keep offsets in memory rather than copying all candidate memberships."""

    def __init__(self, path):
        self.path, self.by_source = Path(path), {}
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                if row["candidates"]:
                    self.by_source.setdefault(row["source"], []).append(offset)
        if not self.by_source:
            raise ValueError("No nonempty set-training episodes")

    def read(self, offset, stream):
        stream.seek(offset)
        return json.loads(stream.readline())

    def sample(self, rng, repeats):
        names = sorted(self.by_source)
        for _ in range(repeats):
            for name in rng.permutation(names):
                offsets = self.by_source[name]
                yield offsets[int(rng.integers(len(offsets)))]


@register("scorer", "set_model")
class SetScorer:
    uses_features = False

    def __init__(
        self,
        objective="compatibility",
        width=256,
        queries=4,
        heads=4,
        layers=2,
        feedforward=512,
        max_epochs=100,
        batch_size=32,
        repeats_per_source=24,
        learning_rate=0.001,
        temperature=0.1,
        patience=10,
        validation_fraction=0.2,
        max_batch_values=8000000,
        device="cpu",
        seed=73019,
        validation_strategy=None,
    ):
        integer_values = [
            width,
            queries,
            heads,
            layers,
            feedforward,
            max_epochs,
            batch_size,
            repeats_per_source,
            patience,
            max_batch_values,
        ]
        if any(type(v) is not int or v < 1 for v in integer_values) or width % heads:
            raise ValueError("Invalid set network dimensions or training counts")
        if objective not in ["compatibility", "completion", "matched_completion"] or device not in [
            "cpu",
            "cuda",
        ]:
            raise ValueError("Invalid set objective/device")
        if not 0 <= validation_fraction < 1 or learning_rate <= 0 or temperature <= 0:
            raise ValueError("Invalid set optimization parameters")
        self.objective, self.width, self.queries = objective, width, queries
        self.heads, self.layers, self.feedforward = heads, layers, feedforward
        self.max_epochs, self.batch_size, self.repeats_per_source = max_epochs, batch_size, repeats_per_source
        self.learning_rate, self.temperature, self.patience = learning_rate, temperature, patience
        self.validation_fraction, self.max_batch_values, self.device, self.seed = (
            validation_fraction,
            max_batch_values,
            device,
            seed,
        )
        if validation_strategy is not None:
            if not isinstance(validation_strategy, dict) or set(validation_strategy) - {"name", "params"}:
                raise ValueError("Invalid validation strategy specification")
            build("validation_strategy", validation_strategy)
        self.validation_strategy = validation_strategy
        self.state_dict, self.dimension, self.history = None, None, []

    def _values(self, context):
        value = representation_values(context.recipe["scorer"], context.representation)
        embedding, coverage = value["embedding"], value["coverage"]
        if embedding.ndim != 2 or coverage.shape != (len(context.store.ids), 1) or (coverage < 0).any():
            raise ValueError("Set scorer needs embeddings and scalar raw, nonnegative coverage")
        return embedding, coverage[:, 0]

    def _batch(self, records, embeddings, coverage):
        import torch

        mc, mn = max(len(r["core"]) for r in records), max(len(r["candidates"]) for r in records)
        d, n = embeddings.shape[1], len(records)
        if n * (mc + mn) * d > self.max_batch_values:
            raise ValueError("Set batch exceeds max_batch_values; explicitly reduce batch_size")
        estimated = n * (mc + mn) * d * 32 + n * self.heads * (mc * mc + self.queries * mc) * 16
        if estimated > getattr(self, "_memory_bytes", 8 * 1024**3):
            raise ValueError("Set attention/batch exceeds the declared memory budget")
        core = np.zeros((n, mc, d), np.float32)
        candidates = np.zeros((n, mn, d), np.float32)
        mask = np.zeros((n, mc), bool)
        cm = np.zeros((n, mn), bool)
        labels = np.zeros_like(cm)
        cov = np.zeros(mask.shape, np.float32)
        for b, row in enumerate(records):
            c, cc = row["core"], row["candidates"]
            core[b, : len(c)], candidates[b, : len(cc)] = embeddings[c], embeddings[cc]
            mask[b, : len(c)] = True
            cm[b, : len(cc)] = True
            labels[b, : len(cc)] = row["labels"]
            lc = np.log(coverage[c] + 0.1)
            cov[b, : len(c)] = lc - lc.mean()
        return [torch.as_tensor(v, device=self.device) for v in [core, mask, cov, candidates, labels, cm]]

    def _network(self):
        from .set_network import BinCompletion

        return BinCompletion(
            self.dimension, self.width, self.queries, self.heads, self.layers, self.feedforward
        )

    def _validation(self, model, bank, embeddings, coverage):
        if getattr(self, "_selection_validator", None) is not None:
            return self._selection_validator.evaluate(
                model, bank, embeddings, coverage, self._selection_context
            )
        import torch

        from ..pipeline.examples import iter_records

        scores = {}
        model.eval()
        with torch.inference_mode():
            for row in iter_records(bank.path):
                if row["absent"] or not row["hidden_total"]:
                    continue
                if not row["candidates"]:
                    recall = 0.0
                else:
                    c, mask, cov, x, y, cm = self._batch([row], embeddings, coverage)
                    s = (model(c, mask, cov)[0] @ x[0].T).amax(0).cpu().numpy()
                    labels = np.asarray(row["labels"], bool)
                    # Pessimistic ties: tied candidates count ahead of a target.
                    recall = sum((s >= s[p]).sum() <= 5 for p in np.flatnonzero(labels)) / row["hidden_total"]
                scores.setdefault(row["source"], []).append(float(recall))
        return float(np.mean([np.mean(v) for v in scores.values()])) if scores else 0.0

    def _optimize(self, context, bank, embeddings, coverage, folder, epochs, validation=None):
        import torch

        from .set_network import set_loss

        torch.manual_seed(self.seed)
        model = self._network().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        start = 0
        history = []
        best = -1.0
        stale = 0
        best_epoch = 1
        checkpoint = folder / "epoch.pt"
        freeze(
            folder / "design.json",
            dict(
                examples=str(bank.path),
                epochs=epochs,
                objective=self.objective,
                scoring_signature=context.scoring_signature,
                validation=str(validation.path) if validation else None,
            ),
        )
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            if self.device == "cuda" and saved.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            start, history, best, stale, best_epoch = (
                saved["epoch"],
                saved["history"],
                saved["best"],
                saved["stale"],
                saved["best_epoch"],
            )
        for epoch in range(start, epochs):
            if validation is not None and stale >= self.patience:
                break
            model.train()
            losses = []
            pending = []
            with bank.path.open("rb") as stream:
                offsets = iter(bank.sample(rng, self.repeats_per_source))
                while True:
                    pending = []
                    for _ in range(self.batch_size):
                        offset = next(offsets, None)
                        if offset is None:
                            break
                        pending.append(bank.read(offset, stream))
                    if not pending:
                        break
                    c, mask, cov, x, y, cm = self._batch(pending, embeddings, coverage)
                    optimizer.zero_grad(set_to_none=True)
                    loss = set_loss(model(c, mask, cov), x, y, cm, self.objective, self.temperature)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite set loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            metric = self._validation(model, validation, embeddings, coverage) if validation else None
            history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), validation_recall_at_5=metric))
            if metric is not None:
                if metric > best + 1e-6:
                    best, stale, best_epoch = metric, 0, epoch + 1
                else:
                    stale += 1
            temporary = checkpoint.with_suffix(".partial.pt")
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if self.device == "cuda" else None,
                    epoch=epoch + 1,
                    history=history,
                    best=best,
                    stale=stale,
                    best_epoch=best_epoch,
                ),
                temporary,
            )
            temporary.replace(checkpoint)
            context.status(
                "set_training",
                objective=self.objective,
                epoch=epoch + 1,
                epochs=epochs,
                validation_recall_at_5=metric,
                phase_directory=folder.name,
            )
        return model, history, best_epoch

    def fit(self, context, example_path, stats, folder, sources, groups):
        import torch

        from ..config.references import references
        from ..pipeline.examples import example_artifact, iter_records

        for row in iter_records(example_path):
            for name in references(context.recipe["scorer"]).values():
                context.require_representation_rows(name, np.asarray(row["core"] + row["candidates"], int))
        embeddings, coverage = self._values(context)
        self.dimension = embeddings.shape[1]
        self._memory_bytes = context.recipe["resources"]["memory_gib"] * 1024**3 - getattr(
            context, "reserved_bytes", 0
        )
        torch_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, context.recipe["resources"]["threads"]))
        self._selection_context = context
        self._selection_validator = (
            build("validation_strategy", self.validation_strategy) if self.validation_strategy else None
        )
        epochs = self.max_epochs
        selection = None
        try:
            if self.validation_fraction:
                names = sorted({groups[r["genome_id"]] for r in sources})
                if len(names) < 4:
                    raise ValueError(
                        "Set epoch selection needs >=4 groups; explicitly set validation_fraction=0 for fixed epochs"
                    )
                rng = np.random.default_rng(self.seed)
                nval = max(2, int(np.ceil(len(names) * self.validation_fraction)))
                held = set(rng.permutation(names)[: min(nval, len(names) - 2)])
                fit_sources = [r for r in sources if groups[r["genome_id"]] not in held]
                val_sources = [r for r in sources if groups[r["genome_id"]] in held]
                train_path, _ = example_artifact(context, fit_sources, groups)
                val_path, _ = example_artifact(context, val_sources, groups, evaluation=True)
                selection = dict(
                    train_sources=[r["genome_id"] for r in fit_sources],
                    validation_sources=[r["genome_id"] for r in val_sources],
                    validation_groups=sorted(held),
                )
                freeze(folder / "selection.json", selection)
                _, selection_history, epochs = self._optimize(
                    context,
                    EpisodeBank(train_path),
                    embeddings,
                    coverage,
                    folder / "selection",
                    self.max_epochs,
                    EpisodeBank(val_path),
                )
                save(folder / "selection_history.json", selection_history)
            model, self.history, _ = self._optimize(
                context, EpisodeBank(example_path), embeddings, coverage, folder / "refit", epochs
            )
            self.state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save(folder / "history.json", self.history)
            save(
                folder / "summary.json",
                dict(
                    rows=stats["rows"],
                    sources=stats["sources"],
                    objective=self.objective,
                    feature_schema=[],
                    epochs=epochs,
                    inner_group_selection=selection,
                    full_source_refit=True,
                    equal_source_sampling=True,
                    learned_encoder=False,
                    examples_sha256=stats["sha256"],
                ),
            )
            return self
        finally:
            self.__dict__.pop("_selection_context", None)
            self.__dict__.pop("_selection_validator", None)
            torch.set_num_threads(torch_threads)

    def predict_queries(self, core, context):
        import torch

        from ..core import QuerySet

        embeddings, coverage = self._values(context)
        if not len(core):
            raise ValueError("Cannot predict from an empty core")
        model = self._network().to(self.device)
        model.load_state_dict(self.state_dict)
        model.eval()
        c = torch.as_tensor(np.asarray(embeddings[core], np.float32), device=self.device).unsqueeze(0)
        mask = torch.ones(c.shape[:2], dtype=torch.bool, device=self.device)
        lc = np.log(coverage[core] + 0.1)
        lc -= lc.mean()
        with torch.inference_mode():
            queries = model(
                c, mask, torch.as_tensor(lc, dtype=torch.float32, device=self.device).unsqueeze(0)
            )[0]
        name = context.recipe["scorer"]["representations"]["embedding"]
        return QuerySet(queries.cpu().numpy(), context.representation_id(name))

    def _score(self, core, candidates, context):
        if not len(candidates):
            return np.array([], float)
        import torch

        embeddings, coverage = self._values(context)
        model = self._network().to(self.device)
        model.load_state_dict(self.state_dict)
        model.eval()
        c = torch.as_tensor(np.asarray(embeddings[core], np.float32), device=self.device).unsqueeze(0)
        mask = torch.ones(c.shape[:2], dtype=torch.bool, device=self.device)
        lc = np.log(coverage[core] + 0.1)
        lc -= lc.mean()
        with torch.inference_mode():
            queries = model(
                c, mask, torch.as_tensor(lc, dtype=torch.float32, device=self.device).unsqueeze(0)
            )[0]
            result = np.empty(len(candidates), float)
            size = context.recipe["resources"]["chunk_size"]
            for a in range(0, len(candidates), size):
                x = torch.as_tensor(
                    np.asarray(embeddings[candidates[a : a + size]], np.float32), device=self.device
                )
                # Recovery used raw max cosine; the 1/temperature scaling is for the loss.
                result[a : a + len(x)] = (x @ queries.T).amax(1).cpu().numpy()
        return result

    def score(self, core, candidates, context):
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(min(2, context.recipe["resources"]["threads"]))
        try:
            return self._score(core, candidates, context)
        finally:
            torch.set_num_threads(previous)
