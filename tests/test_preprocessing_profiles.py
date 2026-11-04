"""Explicit preprocessing and source identities are experiment inputs."""

from itertools import product

import numpy as np
import pytest

from bincomplete import prepare_dataset
from bincomplete.context import Context
from bincomplete.methods import recipe
from bincomplete.representations.kmers import frequencies


def test_count_basis_pseudocount_and_float32_normalization():
    sequence = "AAAATGCATGCATTTNNNCCCCGGG"
    mapping = {}
    for word in map("".join, product("ATGC", repeat=4)):
        if word not in mapping:
            value = len(set(mapping.values()))
            mapping[word] = value
            mapping[word.translate(str.maketrans("ACGT", "TGCA"))[::-1]] = value
    values = np.bincount(
        [mapping[sequence[i : i + 4]] for i in range(len(sequence) - 3) if sequence[i : i + 4] in mapping],
        minlength=136,
    ).astype(float)
    values += 1e-5
    values = (values / values.sum()).astype(np.float32)
    values /= np.cumsum(values, dtype=np.float32)[-1]
    np.testing.assert_array_equal(
        frequencies(
            sequence.encode(), 4, True, alphabet_order="ATGC", pseudocount=1e-5, renormalize_float32=True
        ),
        values,
    )


def test_original_source_ids_and_groups_control_examples(dataset, tmp_path, limits):
    parent = dataset.path.parent
    identities = tmp_path / "source_ids.tsv"
    groups = tmp_path / "groups.tsv"
    identities.write_text("bin_id\tsource_id\n" + "".join(f"b{i}\tstable{i}\n" for i in range(4)))
    groups.write_text("bin_id\tgroup_id\n" + "".join(f"b{i}\tgroup{i}\n" for i in range(4)))
    prepared = prepare_dataset(
        tmp_path / "mapped",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        source_ids=identities,
        groups=groups,
    )
    context = Context(
        prepared,
        recipe("kmers", limits, {"preprocessing": {"coverage_dtype": "float32"}}),
        tmp_path / "work",
        tmp_path / "cache",
        limits,
    )
    try:
        sources, relations = context.store.sources()
        assert [r["genome_id"] for r in sources] == [f"stable{i}" for i in range(4)]
        assert relations["stable0"] == "group0"
        assert context.representation("coverage_features").dtype == np.float32
    finally:
        context.close()


def test_token_pieces_preserve_bases_and_bound_tokens():
    from bincomplete.representations.strand_pooling import token_pieces

    def tokenizer(sequence, **kwargs):
        return {"input_ids": [1] + [2] * len(sequence) + [3]}

    pieces = token_pieces("ACGT" * 11, tokenizer, 13)
    assert "".join(pieces) == "ACGT" * 11
    assert all(len(piece) + 2 <= 13 for piece in pieces)


def test_strand_pooling_excludes_special_tokens_and_normalizes():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from bincomplete.representations.strand_pooling import pooled_vectors

    class Tokenizer:
        all_special_ids = [0, 9]

        def __call__(self, sequences, **kwargs):
            if isinstance(sequences, str):
                return {"input_ids": [9] + [1] * len(sequences) + [9]}
            n = max(map(len, sequences)) + 2
            ids = torch.zeros(len(sequences), n, dtype=torch.int64)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(sequences):
                ids[i, 0] = 9
                ids[i, len(s) + 1] = 9
                ids[i, 1 : len(s) + 1] = 1
                mask[i, : len(s) + 2] = 1
            return {"input_ids": ids, "attention_mask": mask}

    class Model:
        def __call__(self, **values):
            v = values["input_ids"].float()
            return SimpleNamespace(last_hidden_state=torch.stack([v, torch.ones_like(v)], -1))

    space = {"window_bp": 6, "windows": 4, "reverse_complement": True, "max_tokens": 6}
    actual = pooled_vectors(
        Tokenizer(), Model(), [b"ACGT" * 4], space, {"device": "cpu", "batch_size": 3, "precision": "float32"}
    )
    np.testing.assert_allclose(actual, [[1 / np.sqrt(2), 1 / np.sqrt(2)]], atol=1e-7)


def test_supplied_original_composition_preserves_zeros_and_row_alignment(dataset, tmp_path, limits):
    import csv

    parent = dataset.path.parent
    ids = np.load(dataset.path / "ids.npy", allow_pickle=False)
    source = tmp_path / "composition.csv"
    expected = {}
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["contig_id", *map(str, range(136)), "ignored_quality"])
        for number, name in reversed(list(enumerate(ids))):
            values = np.linspace(0.001, 0.02, 136).astype(np.float32)
            values[number % 136] = 0
            expected[str(name)] = values / np.cumsum(values, dtype=np.float32)[-1]
            writer.writerow([name, *map(str, values), "not_read"])
    prepared = prepare_dataset(
        tmp_path / "supplied",
        assembly=parent / "assembly.fa",
        memberships=parent / "members.tsv",
        coverage=parent / "coverage.tsv",
        composition=source,
    )
    settings = {"preprocessing": {"composition": {"renormalize_float32": True}}}
    context = Context(
        prepared, recipe("kmers", limits, settings), tmp_path / "work", tmp_path / "cache", limits
    )
    try:
        actual = context.representation("kmers")
        for row, name in enumerate(ids):
            np.testing.assert_array_equal(actual[row], expected[str(name)])
        assert context.compatibility()["composition"]["formula"] == "supplied_table_normalized"
    finally:
        context.close()
