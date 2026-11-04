import inspect

import numpy as np

from ..artifacts import fingerprint, sha
from ..config.references import references
from ..config.references import values as representation_values
from ..core import FeatureBatch, build


class FeatureComposer:
    def __init__(
        self,
        specs,
        representation_getter,
        chunk_size=4096,
        representation_identity=None,
        allow_empty=False,
        row_validator=None,
    ):
        self.blocks = [
            (key, spec, build("feature", spec))
            for key, spec in sorted(specs.items())
            if spec.get("enabled", True)
        ]
        if not self.blocks and not allow_empty:
            raise ValueError("At least one classifier feature is required")
        self.get_representation, self.chunk_size = representation_getter, chunk_size
        self.representation_identity = representation_identity
        self.row_validator = row_validator
        self.schema = tuple(
            key + "." + column for key, _, block in self.blocks for column in block.column_names()
        )
        self.identity = fingerprint({key: spec for key, spec, _ in self.blocks})

    def transform(self, core, candidates, store):
        core, candidates = np.asarray(core, int), np.asarray(candidates, int)
        if (
            not len(core)
            or len(set(core)) != len(core)
            or len(set(candidates)) != len(candidates)
            or set(core) & set(candidates)
        ):
            raise ValueError("Invalid core/candidate membership or feature leakage")
        output = np.empty((len(candidates), len(self.schema)), np.float32)
        column = 0
        for _, spec, block in self.blocks:
            if self.row_validator:
                for name in references(spec).values():
                    self.row_validator(name, np.r_[core, candidates])
            values = representation_values(spec, self.get_representation)
            identity = dict(
                block=spec,
                metadata=store.input_hash("features" if "features" in store.inputs else "contigs"),
                representation={
                    key: self.representation_identity(name) if self.representation_identity else name
                    for key, name in references(spec).items()
                },
                implementation=sha(inspect.getfile(type(block))),
                core=core.tolist(),
                candidates=candidates.tolist(),
            )
            transformed = store.cache.feature(
                identity, lambda: block.transform(core, candidates, values, store.lengths, self.chunk_size)
            )
            width = len(block.column_names())
            if transformed.shape != (len(candidates), width) or not np.isfinite(transformed).all():
                raise ValueError("Feature implementation returned an invalid matrix")
            output[:, column : column + width] = transformed
            column += width
        return FeatureBatch(output, self.schema)
