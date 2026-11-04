import numpy as np
from threadpoolctl import threadpool_limits

from ..artifacts import save
from ..core import register


@register("scorer", "feature_classifier")
class FeatureClassifier:
    uses_features = True

    def __init__(self):
        self.classifier = None

    def fit(self, context, example_path, stats, folder, sources, groups):
        from ..pipeline.examples import iter_records

        n, d = stats["rows"], len(context.features.schema)
        if n == 0 or stats["positives"] in [0, n]:
            raise ValueError("Sampling/retrieval produced no mixed-label training set")
        estimated_bytes = n * (12 * d + 96) + getattr(context, "reserved_bytes", 0)
        if estimated_bytes > context.recipe["resources"]["memory_gib"] * 1024**3:
            raise ValueError("Estimated training memory exceeds budget; change the sampling/model explicitly")
        context.status("training_features", completed_rows=0, total_rows=n)
        arrays = {
            name: np.lib.format.open_memmap(
                folder / (name + ".partial.npy"), mode="w+", dtype=dtype, shape=shape
            )
            for name, dtype, shape in [
                ("features", "float32", (n, d)),
                ("labels", "uint8", (n,)),
                ("weights", "float64", (n,)),
            ]
        }
        start = 0
        weighting = context.recipe["training"].get("weighting", "source_balanced")
        for episode in iter_records(example_path):
            cc = np.array(episode["candidates"], int)
            labels = np.array(episode["labels"], np.uint8)
            if not len(cc):
                continue
            end = start + len(cc)
            arrays["features"][start:end] = context.features.transform(
                np.array(episode["core"], int), cc, context.store
            ).values
            arrays["labels"][start:end] = labels
            positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
            mass = stats["source_mass"][episode["source"]] if weighting == "source_balanced" else 1.0
            arrays["weights"][start:end] = (
                np.where(labels, 1 / max(1, positives), 1 / max(1, negatives)) / mass
            )
            start = end
        if start != n:
            raise ValueError("Example row count changed during training")
        for array in arrays.values():
            array.flush()
        context.status("fitting", completed_rows=n, total_rows=n, estimated_training_bytes=estimated_bytes)
        with threadpool_limits(limits=context.recipe["resources"]["threads"]):
            self.classifier = context.components["classifier"].fit(
                arrays["features"], arrays["labels"], sample_weight=arrays["weights"]
            )
        for array in arrays.values():
            array._mmap.close()
        arrays.clear()
        del array
        for name in ["features", "labels", "weights"]:
            (folder / (name + ".partial.npy")).replace(folder / (name + ".npy"))
        save(
            folder / "summary.json",
            dict(
                rows=n,
                positives=stats["positives"],
                sources=stats["sources"],
                feature_schema=list(context.features.schema),
                estimated_training_bytes=estimated_bytes,
                equal_source_weight=weighting == "source_balanced",
                weighting=weighting,
                examples_sha256=stats["sha256"],
                classifier=context.recipe["classifier"],
            ),
        )
        return self

    def score(self, core, candidates, context):
        return self.classifier.predict_scores(
            context.features.transform(core, candidates, context.store).values
        )
