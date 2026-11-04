"""Split-local supervised examples, streamed to a reusable immutable JSONL artifact."""

import json
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..artifacts import fingerprint, read, save, sha
from ..config.references import references
from ..core import Episode, build
from ..masking import source_rng


@dataclass
class Example:
    episode: Episode
    candidates: np.ndarray
    labels: np.ndarray
    hidden_total: int
    hidden_retrieved: int


def examples(context, sources, groups, evaluation=False):
    store = context.store
    bins = {r["genome_id"]: store.indices(r["members"]) for r in sources}
    seed = context.recipe["seed"]
    for record in sources:
        gid = record["genome_id"]
        members = bins[gid]
        negative_bins = [rows for other, rows in bins.items() if groups[other] != groups[gid]]
        if len(members) < 2 or not negative_bins:
            continue
        negatives = np.unique(np.concatenate(negative_bins))
        validation = context.recipe.get("validation", {})
        masker = (
            build("masker", validation["masker"])
            if evaluation and validation.get("masker")
            else context.components["masker"]
        )
        generated = masker.generate(members, store.lengths, source_rng(seed, "mask:" + gid))
        for number, mask in enumerate(generated):
            if (
                not len(mask.core)
                or not len(mask.hidden)
                or len(set(mask.core)) != len(mask.core)
                or len(set(mask.hidden)) != len(mask.hidden)
                or set(mask.core) & set(mask.hidden)
                or set(mask.core) | set(mask.hidden) != set(members)
            ):
                raise ValueError(
                    "Masker must partition only its source members into disjoint nonempty core/hidden sets"
                )
            episode = Episode(gid, groups[gid], mask.core, mask.hidden, mask.absent)
            truth = np.array([], int) if mask.absent else mask.hidden
            sc = SimpleNamespace(negatives=negatives, store=store, context=context)
            positive = (
                truth
                if evaluation
                else np.asarray(
                    context.components["positive_sampler"].sample(
                        episode, sc, source_rng(seed, f"positive:{gid}:{number}")
                    ),
                    int,
                )
            )
            if len(set(positive)) != len(positive) or not set(positive) <= set(truth):
                raise ValueError("Positive sampler returned something other than hidden source members")
            universe = np.unique(np.r_[truth, negatives])
            all_known = (
                validation.get("candidate_policy", "retrieved")
                if evaluation
                else context.recipe["training"].get("candidate_policy", "retrieved")
            ) == "all_known"
            retrieved = universe if all_known else context.retrieve(mask.core, universe).indices
            if evaluation:
                candidates = retrieved
            else:
                sc.retrieved = retrieved
                sc.eligible_negatives = (
                    context.components["retriever"].eligible(mask.core, negatives, store)
                    if getattr(context.components["negative_sampler"], "needs_eligibility", True)
                    else negatives
                )
                negative = np.asarray(
                    context.components["negative_sampler"].sample(
                        episode, sc, source_rng(seed, f"negative:{gid}:{number}")
                    ),
                    int,
                )
                if len(set(negative)) != len(negative) or not set(negative) <= set(negatives):
                    raise ValueError("Negative sampler leaked unknown, related or source contigs")
                # Do not inject positives that retrieval could not find.
                candidates = np.unique(np.r_[positive[np.isin(positive, retrieved)], negative])
            candidates = candidates[np.argsort(store.ids[candidates], kind="stable")]
            labels = np.isin(candidates, positive).astype(np.uint8)
            yield Example(episode, candidates, labels, len(truth), int(np.isin(truth, candidates).sum()))


def example_artifact(context, sources, groups, evaluation=False):
    spec = context.recipe
    candidate_policy = (
        spec["validation"]["candidate_policy"]
        if evaluation
        else spec["training"].get("candidate_policy", "retrieved")
    )
    depends_on_retrieval = candidate_policy != "all_known" or (
        not evaluation and getattr(context.components["negative_sampler"], "needs_retrieval", True)
    )
    refs = references(spec["retriever"])
    narrow_names = getattr(context.components["retriever"], "narrow_input_names", set(refs))
    refs = {k: v for k, v in refs.items() if k in narrow_names}
    identity = dict(
        kind="split_local_examples_v1",
        data=context.store.input_hash("training_bins"),
        metadata=context.store.input_hash("features" if "features" in context.store.inputs else "contigs"),
        coverage=context.store.array_identity("coverage")
        if depends_on_retrieval
        and getattr(context.components["retriever"], "coverage_fold", None) is not None
        else None,
        groups={r["genome_id"]: groups[r["genome_id"]] for r in sources},
        members=[(r["genome_id"], r["members"]) for r in sources],
        representation={key: context.representation_id(name) for key, name in refs.items()}
        if depends_on_retrieval
        else None,
        fitted_retriever=getattr(context.components["retriever"], "state_identity", None)
        if depends_on_retrieval
        else None,
        candidate_policy=candidate_policy,
        seed=spec["seed"],
        masker=(spec["validation"]["masker"] or spec["masker"]) if evaluation else spec["masker"],
        retriever=dict(
            name=spec["retriever"]["name"], params=spec["retriever"].get("params", {}), representations=refs
        )
        if depends_on_retrieval
        else None,
        evaluation=evaluation,
        positive_sampler=None if evaluation else spec["positive_sampler"],
        negative_sampler=None if evaluation else spec["negative_sampler"],
        negative_representations={
            key: context.representation_id(name) for key, name in references(spec["negative_sampler"]).items()
        }
        if not evaluation
        else None,
        code={
            k: v
            for k, v in context.code.items()
            if k.startswith(("sampling", "retrieval", "pipeline/examples"))
        },
    )
    folder = context.output / "examples" / fingerprint(identity)
    marker, destination = folder / "complete.json", folder / "examples.jsonl"
    if marker.exists():
        proof = read(marker)
        if (
            fingerprint(proof["identity"]) != fingerprint(identity)
            or not destination.exists()
            or sha(destination) != proof["sha256"]
        ):
            raise ValueError("Example checkpoint integrity changed")
        if not evaluation and proof["rows"] > spec["training"]["max_training_rows"]:
            raise ValueError("Cached examples exceed the explicitly declared training row budget")
        context.cache.hits += 1
        return destination, proof
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "examples.partial.jsonl"
    count = positives = episodes = 0
    mass = Counter()
    sources_seen = set()
    with temporary.open("w", encoding="utf-8") as stream:
        for example in examples(context, sources, groups, evaluation):
            ep = example.episode
            labels = example.labels
            record = dict(
                source=ep.source,
                group=ep.group,
                core=ep.core.tolist(),
                hidden=ep.hidden.tolist(),
                absent=ep.absent,
                candidates=example.candidates.tolist(),
                labels=labels.tolist(),
                hidden_total=example.hidden_total,
                hidden_retrieved=example.hidden_retrieved,
            )
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            npos = int(labels.sum())
            nneg = len(labels) - npos
            mass[ep.source] += int(npos > 0) + int(nneg > 0)
            count += len(labels)
            positives += npos
            episodes += 1
            if len(labels):
                sources_seen.add(ep.source)
            if not evaluation and count > spec["training"]["max_training_rows"]:
                raise ValueError("Training row budget exceeded; change sampling explicitly, not silently")
            if episodes % 100 == 0:
                context.status(
                    "validation_examples" if evaluation else "training_examples",
                    completed_episodes=episodes,
                    completed_rows=count,
                )
    temporary.replace(destination)
    proof = dict(
        identity=identity,
        sha256=sha(destination),
        rows=count,
        positives=positives,
        episodes=episodes,
        sources=len(sources_seen),
        source_mass=dict(mass),
    )
    save(marker, proof)
    return destination, proof


def iter_records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)
