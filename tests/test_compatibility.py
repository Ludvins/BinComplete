import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bincomplete.compat import forward_legacy
from bincomplete.features.membership import Composition, Coverage, EmbeddingSimilarity
from bincomplete.search.ranked_alternatives import RankedAlternatives


def test_native_legacy_commands_cannot_be_forwarded(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected process"))
    for command in ("score", "recover", "study", "--native"):
        with pytest.raises(ValueError, match="external"):
            forward_legacy(["--config", "example.json", command])


def test_numerical_formulas_and_proposal_order_match_preserved_code(tmp_path):
    configured = os.environ.get("BINCOMPLETE_GOLDEN_ROOT") or os.environ.get("BIN_RECRUITMENT_GOLDEN_ROOT")
    if configured is None:
        configured = str(Path(__file__).resolve().parents[1] / "provenance/golden")
    root = Path(configured)
    feature_module = os.environ.get("BINCOMPLETE_GOLDEN_FEATURE_MODULE", "bincomplete.features.membership")
    search_module = os.environ.get(
        "BINCOMPLETE_GOLDEN_SEARCH_MODULE", "bincomplete.search.ranked_alternatives"
    )
    if not (root / "src" / Path(*feature_module.split("."))).with_suffix(".py").is_file():
        pytest.skip("Provide a preserved legacy source tree for cross-version golden checks")
    rng = np.random.default_rng(73)
    k = rng.random((24, 136)).astype("float32")
    k /= k.sum(1, keepdims=True)
    z = rng.normal(size=(24, 8)).astype("float32")
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    coverage = rng.uniform(1, 15, (24, 2))
    lengths = np.full(24, 2000)
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, k=k, z=z, coverage=coverage, lengths=lengths)
    program = """
import json,sys,numpy as np
from types import SimpleNamespace
import importlib
module=importlib.import_module(sys.argv[3])
Composition,Coverage,EmbeddingSimilarity=module.Composition,module.Coverage,module.EmbeddingSimilarity
RankedAlternatives=importlib.import_module(sys.argv[4]).RankedAlternatives
x=np.load(sys.argv[1]); core=np.array([0,1,2]); cc=np.arange(3,24)
features=[b.transform(core,cc,x[key],x['lengths'],7) for b,key in [(Composition(),'k'),(Coverage(),'coverage'),(EmbeddingSimilarity(),'z')]]
np.savez(sys.argv[2], **{str(i):value for i,value in enumerate(features)})
store=SimpleNamespace(ids=np.array([str(i) for i in range(24)]), lengths=x['lengths'], protected=lambda:set())
print(json.dumps(list(RankedAlternatives().propose(core,cc,store)),sort_keys=True,default=int))
"""
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    previous = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            program,
            str(inputs),
            str(tmp_path / "old.npz"),
            feature_module,
            search_module,
        ],
        env=environment,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    old = np.load(tmp_path / "old.npz")
    core, candidates = np.array([0, 1, 2]), np.arange(3, 24)
    for i, (block, values) in enumerate(
        ((Composition(), k), (Coverage(), coverage), (EmbeddingSimilarity(), z))
    ):
        np.testing.assert_array_equal(block.transform(core, candidates, values, lengths, 7), old[str(i)])
    import json

    store = SimpleNamespace(
        ids=np.array([str(i) for i in range(24)]), lengths=lengths, protected=lambda: set()
    )
    assert list(RankedAlternatives().propose(core, candidates, store)) == json.loads(previous.stdout)
