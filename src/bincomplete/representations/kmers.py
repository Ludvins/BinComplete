import numpy as np


def frequencies(
    sequence, k, canonical=True, *, alphabet_order="ACGT", pseudocount=0.0, renormalize_float32=False
):
    """Compute canonical frequencies with an explicit basis and precision policy."""
    if alphabet_order not in {"ACGT", "ATGC"} or not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("Invalid composition basis or pseudocount")
    alphabet = np.full(256, -1, dtype=np.int8)
    alphabet[np.frombuffer(alphabet_order.encode(), dtype=np.uint8)] = np.arange(4)
    bases = alphabet[np.frombuffer(sequence.upper(), dtype=np.uint8)]
    size = 4**k
    codes = np.arange(size, dtype=np.int64)
    reverse = np.zeros(size, np.int64)
    work = codes.copy()
    for _ in range(k):
        complement = np.array(
            [alphabet_order.index(c) for c in alphabet_order.translate(str.maketrans("ACGT", "TGCA"))]
        )
        reverse = reverse * 4 + complement[work % 4]
        work //= 4
    mapping = np.minimum(codes, reverse) if canonical else codes
    _, mapping = np.unique(mapping, return_inverse=True)
    counts = np.zeros(int(mapping.max()) + 1, float)
    # Keep temporary sequence windows bounded even for long contigs.
    for start in range(0, max(0, len(bases) - k + 1), 65536):
        length = min(65536, len(bases) - k + 1 - start)
        code = np.zeros(length, np.int64)
        valid = np.ones(length, bool)
        for offset in range(k):
            part = bases[start + offset : start + offset + length]
            valid &= part >= 0
            code = code * 4 + np.maximum(part, 0)
        counts += np.bincount(mapping[code[valid]], minlength=len(counts))
    counts += pseudocount
    if not counts.sum():
        raise ValueError("Contig has no valid k-mers")
    values = (counts / counts.sum()).astype("float32")
    if renormalize_float32:
        # Match a column-major feature table's sequential row reduction.
        values /= np.cumsum(values, dtype=np.float32)[-1]
    return values
