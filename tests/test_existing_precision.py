"""Read-only integration contract for frozen reduced-precision embeddings."""

import numpy as np

from bincomplete import prepare_dataset
from bincomplete.artifacts import sha
from bincomplete.context import Context, RowView
from bincomplete.methods import recipe


def test_precomputed_float16_is_used_in_place(dataset, tmp_path, limits):
    parent = dataset.path.parent
    path = tmp_path / "saved_float16.npy"
    np.save(path, np.load(parent / "embedding.npy").astype(np.float16))
    before = sha(path)
    prepared = prepare_dataset(
        tmp_path / "prepared_half",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        embeddings=path,
        embedding_ids=parent / "embedding_ids.txt",
        embedding_space=parent / "space.json",
    )
    assert prepared.manifest["embedding"]["storage_dtype"] == "float16"
    context = Context(prepared, recipe("cosine", limits), tmp_path / "work", tmp_path / "cache", limits)
    try:
        values = context.representation("embedding")
        assert isinstance(values, RowView)
        assert isinstance(values.matrix, np.memmap)
        assert values.matrix.dtype == np.float16
        np.testing.assert_array_equal(values[[0]][0], np.load(path)[-1])
    finally:
        context.close()
    assert sha(path) == before
    assert not (prepared.path / "embedding.npy").exists()
