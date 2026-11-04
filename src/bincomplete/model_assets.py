"""Explicit pinned model installation; experiments remain offline/local-only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import read, sha
from .defaults import DNABERT_REPO, DNABERT_REVISION, load_defaults, model_root
from .resources import FileLock


def install_dnabert(
    output: str | Path | None = None,
    *,
    revision: str = DNABERT_REVISION,
    global_config: str | Path | None = None,
) -> dict[str, Any]:
    """Download an explicit checkpoint revision once, or verify existing assets.

    The download includes the upstream model's Python code and weights. Nothing
    is imported from that code, fitted or encoded during installation. Receipts
    live beside the model directory so they do not change its encoder identity.
    Partial/failed installations are retained for inspection and retry.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be a full 40-character commit SHA")
    defaults, _ = load_defaults(global_config)
    if output is None:
        output = (
            defaults["encoder_path"]
            if revision == DNABERT_REVISION
            else model_root() / "dnabert_s" / revision
        )
    target = Path(output).expanduser().resolve()
    if target == Path(target.anchor):
        raise ValueError("Choose a dedicated model directory")
    receipt = target.with_name(target.name + ".download.json")
    attempt = target.with_name(target.name + ".download-attempt.json")
    with FileLock(target.with_name(target.name + ".install.lock"), blocking=False):
        if receipt.exists():
            proof = read(receipt)
            if proof["revision"] != revision:
                raise ValueError("Model directory belongs to another revision")
            for name, digest in proof["files_sha256"].items():
                path = (target / name).resolve()
                if not path.is_relative_to(target) or sha(path) != digest:
                    raise ValueError(f"Installed model changed: {name}")
            from .representations.dna import local_contract

            space, inputs = local_contract(target)
            current = {Path(name).relative_to(target).as_posix(): digest for name, digest in inputs.items()}
            if current != proof["files_sha256"] or space != proof["space"]:
                raise ValueError("Installed model file population or encoder identity changed")
            return dict(proof, reused=True)
        if target.exists() and any(target.iterdir()) and not attempt.exists():
            # A user-managed checkpoint is referenced in place, never overwritten.
            from .representations.dna import local_contract

            space, _ = local_contract(target)
            return {
                "status": "EXISTING_LOCAL_MODEL_REUSED",
                "path": str(target),
                "space": space,
                "revision": None,
                "downloaded": False,
                "reused": True,
            }
        declaration = {"repo_id": DNABERT_REPO, "revision": revision, "path": str(target)}
        if attempt.exists() and read(attempt) != declaration:
            raise ValueError("Incomplete download has a different contract; use a new output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not attempt.exists():
            with attempt.open("x", encoding="utf-8") as stream:
                json.dump(declaration, stream, indent=2)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ValueError("Install bincomplete[embeddings] to enable the model installer") from error
        snapshot_download(repo_id=DNABERT_REPO, revision=revision, local_dir=target, max_workers=2)
        from .representations.dna import local_contract

        space, inputs = local_contract(target)
        # No user-supplied files may be injected into a managed download between calls.
        proof = {
            "status": "MODEL_INSTALLED",
            **declaration,
            "space": space,
            "files_sha256": {
                Path(name).relative_to(target).as_posix(): digest for name, digest in inputs.items()
            },
            "downloaded": True,
            "upstream_custom_code": True,
        }
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(proof, stream, indent=2)
            stream.write("\n")
        return dict(proof, reused=False)
