"""External tool configuration, execution and the pilot RNA policy."""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .artifacts import fingerprint, sha

POLICY = "canonical-trna18-v1"
RRNA = {"5S_rRNA", "16S_rRNA", "23S_rRNA"}
CANONICAL_TRNA = frozenset(
    "Ala Arg Asn Asp Cys Gln Glu Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val".split()
)
INTRON_BOUNDS = re.compile(r"[0-9]+(?:,[0-9]+)*")
TOOLS = {
    "checkm2": ("--version", "1.1.0"),
    "barrnap": ("--version", "0.9"),
    "trnascan": ("--help", "2.0.12"),
}
ENV_KEYS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PERL5LIB",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
)


class Paused(Exception):
    """A pause or termination was requested."""


class ToolError(ValueError):
    """An external command did not finish successfully."""

    def __init__(self, tool: str, code: int, stderr: Path):
        super().__init__(f"{tool} exited with code {code}. See {stderr}")
        self.code = code


def _empty_proteins(log: str) -> set[str]:
    return {
        path.replace("\\", "/").rsplit("/", 1)[-1]
        for path in re.findall(r"Skipping protein file (.+?) as it was empty\.", log)
    }


def _unscored() -> dict[str, Any]:
    return dict(status="NO_PREDICTED_PROTEINS", completeness=None, contamination=None)


def read_checkm2_attempt(attempt: Path, names: set[str]) -> dict[str, dict[str, Any]]:
    output = attempt / "results"
    if (output / "quality_report.tsv").is_file():
        return read_checkm2(output, names)
    log = (output / "checkm2.log").read_text(encoding="utf-8")
    stderr = (attempt / "stderr.log").read_text(encoding="utf-8")
    if (
        names
        and _empty_proteins(log) == {f"{name}.faa" for name in names}
        and all(
            (path := output / "protein_files" / f"{name}.faa").is_file() and path.stat().st_size == 0
            for name in names
        )
        and "in __calculate_metadata" in stderr
        and "self.total_threads / len(faa_files)" in stderr
        and stderr.rstrip().endswith("ZeroDivisionError: division by zero")
        and not re.search(r"\b(?:ERROR|CRITICAL):", log)
    ):
        return {name: _unscored() for name in sorted(names)}
    raise ValueError(f"CheckM2 did not produce a verified report. See {attempt / 'stderr.log'}")


def numeric_pass(row: dict[str, Any]) -> bool:
    """Return whether a row passes the numerical screening thresholds."""
    return row["status"] == "SCORED" and row["completeness"] > 90 and row["contamination"] < 5


def full_hq(row: dict[str, Any]) -> bool:
    """Apply the numerical and canonical RNA criteria."""
    return numeric_pass(row) and bool(row["rna"] and row["rna"]["pass"])


def read_checkm2(output: Path, names: set[str]) -> dict[str, dict[str, Any]]:
    report = output / "quality_report.tsv"
    result: dict[str, dict[str, Any]] = {}
    with report.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"Name", "Completeness", "Contamination"} <= set(reader.fieldnames or []):
            raise ValueError(f"Invalid CheckM2 report header: {report}")
        for row in reader:
            name = row["Name"]
            if name not in names or name in result:
                raise ValueError(f"Unexpected or duplicate CheckM2 row: {name}")
            try:
                values = [float(row[key]) for key in ("Completeness", "Contamination")]
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid CheckM2 prediction: {name}") from error
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(f"Invalid CheckM2 prediction: {name}")
            result[name] = dict(status="SCORED", completeness=values[0], contamination=values[1])
    log = (output / "checkm2.log").read_text(encoding="utf-8")
    empty = _empty_proteins(log)
    for name in sorted(names - result.keys()):
        if f"{name}.faa" not in empty:
            raise ValueError(f"CheckM2 result missing for {name}: {report}")
        result[name] = _unscored()
    return result


def read_rna(output: Path) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    rrna: set[str] = set()
    trna: set[str] = set()
    for domain in ("bac", "arc"):
        hits = set()
        labels = set()
        gff = output / f"{domain}.rrna.gff"
        gff_lines = gff.read_text(encoding="utf-8").splitlines()
        if not gff_lines or gff_lines[0].strip() != "##gff-version 3":
            raise ValueError(f"Invalid Barrnap GFF header: {gff}")
        for line in gff_lines:
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 9 or fields[2] != "rRNA":
                raise ValueError(f"Malformed Barrnap record: {gff}")
            attributes = dict(item.split("=", 1) for item in fields[8].split(";") if "=" in item)
            if "Name" not in attributes:
                raise ValueError(f"Missing Barrnap feature name: {gff}")
            hits.add(unquote(attributes["Name"]))
        table = output / f"{domain}.trna.tsv"
        logs = [output / f"{domain}.trnascan.{stream}.log" for stream in ("stdout", "stderr")]
        diagnostics = "\n".join(path.read_text(encoding="utf-8") for path in logs)
        if re.search(r"(?im)^\s*(?:error\s*:|fatal\b)|\bfail(?:ed)? to run infernal\b", diagnostics):
            raise ValueError(f"tRNAscan-SE reported a failed scan. See {logs[0]} and {logs[1]}")
        lines = [line for line in table.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            if not any(
                line.strip() in {"No tRNAs found.", "Status: No tRNAs found."}
                for line in diagnostics.splitlines()
            ):
                raise ValueError(f"Empty tRNAscan-SE table without a no-hit diagnostic: {table}")
        elif (
            len(lines) < 3
            or lines[0].split()[:5] != ["Sequence", "tRNA", "Bounds", "tRNA", "Anti"]
            or lines[1].split()[:4] != ["Name", "tRNA", "#", "Begin"]
            or set("".join(lines[2].split())) != {"-"}
        ):
            raise ValueError(f"Invalid tRNAscan-SE table header: {table}")
        for line in lines[3:]:
            fields = line.split()
            if (
                len(fields) < 9
                or not all(fields[i].isdigit() for i in (1, 2, 3))
                or not all(INTRON_BOUNDS.fullmatch(fields[i]) for i in (6, 7))
                or fields[6].count(",") != fields[7].count(",")
            ):
                raise ValueError(f"Malformed tRNAscan-SE record: {table}")
            try:
                if not math.isfinite(float(fields[8])):
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"Invalid tRNAscan-SE score: {table}") from error
            if fields[4] not in {"Undet", "Sup"}:
                labels.add(fields[4])
        domains[domain] = {"rrna": sorted(hits & RRNA), "trna_labels": sorted(labels)}
        rrna.update(hits)
        trna.update(labels)
    canonical = trna & CANONICAL_TRNA
    return {
        "domains": domains,
        "rrna": sorted(rrna & RRNA),
        "trna_labels": sorted(trna),
        "canonical_trna_types": sorted(canonical),
        "pass": RRNA <= rrna and len(canonical) >= 18,
    }


def _positive(raw: dict[str, Any], name: str, default: int) -> int:
    value = raw.get(name, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _file(config: Path, value: str, label: str) -> Path:
    path = (config.parent / Path(value).expanduser()).absolute()
    if not path.is_file():
        raise ValueError(f"{label} must name an existing file: {path}")
    return path


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    item = path.stat()
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns


@dataclass
class Evaluator:
    commands: dict[str, list[str]]
    runtime_files: tuple[Path, ...]
    database: Path
    batch_size: int
    checkm2_threads: int
    rna_threads: int
    solver_seconds: int
    env_keys: tuple[str, ...]
    paused: Callable[[], bool] = lambda: False
    terminating: Callable[[], bool] = lambda: False
    progress: Callable[[str, Path], None] = lambda tool, log: None
    lock_fd: int | None = None
    _environment: dict[str, str] = field(default_factory=dict, init=False)
    _stamps: dict[Path, tuple[int, int, int, int, int]] = field(default_factory=dict, init=False)

    @classmethod
    def load(cls, config: Path) -> Evaluator:
        config = config.resolve()
        with config.open("rb") as stream:
            raw = tomllib.load(stream)
        keys = {
            "commands",
            "runtime_files",
            "runtime_env",
            "checkm2_database",
            "batch_size",
            "checkm2_threads",
            "rna_threads",
            "solver_seconds",
        }
        if raw.keys() - keys:
            raise ValueError(f"Unknown assessment settings: {sorted(raw.keys() - keys)}")
        commands = raw.get("commands")
        if not isinstance(commands, dict) or commands.keys() != TOOLS.keys():
            raise ValueError("Specify [commands] checkm2, barrnap and trnascan as argument arrays")
        resolved = {}
        for name, args in commands.items():
            if not isinstance(args, list) or not args or any(not isinstance(a, str) or not a for a in args):
                raise ValueError(f"Invalid command array: {name}")
            executable = args[0]
            if Path(executable).parent != Path(".") or executable.startswith("."):
                found = str((config.parent / Path(executable).expanduser()).absolute())
            else:
                found = shutil.which(executable)
            if not found or not Path(found).is_file():
                raise ValueError(f"Assessment executable unavailable: {executable}")
            if os.name == "nt" and Path(found).suffix.lower() in {".bat", ".cmd"}:
                raise ValueError("Use an executable or an explicit interpreter for Windows tool commands")
            resolved[name] = [str(Path(found).absolute()), *args[1:]]
        database = raw.get("checkm2_database")
        files = raw.get("runtime_files", [])
        env_keys = raw.get("runtime_env", [])
        if (
            not isinstance(database, str)
            or not database
            or not isinstance(files, list)
            or not files
            or any(not isinstance(file, str) or not file for file in files)
        ):
            raise ValueError("Specify checkm2_database and runtime_files for the toolchain or its images")
        if not isinstance(env_keys, list) or any(
            not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in env_keys
        ):
            raise ValueError("runtime_env must list environment variable names")
        return cls(
            resolved,
            tuple(_file(config, file, "runtime_files") for file in files),
            _file(config, database, "checkm2_database"),
            _positive(raw, "batch_size", 128),
            _positive(raw, "checkm2_threads", 4),
            _positive(raw, "rna_threads", 2),
            _positive(raw, "solver_seconds", 600),
            tuple(sorted(set(env_keys))),
        )

    def identity(self, work: Path) -> dict[str, Any]:
        files = {Path(args[0]) for args in self.commands.values()} | set(self.runtime_files)
        environment = {key: os.environ[key] for key in (*ENV_KEYS, *self.env_keys) if key in os.environ}
        if "PATH" in environment:
            environment["PATH"] = os.pathsep.join(
                str(Path(directory).absolute())
                for directory in environment["PATH"].split(os.pathsep)
                if directory and Path(directory).is_absolute()
            )
        for args in self.commands.values():
            files.update(
                Path(arg)
                for arg in args[1:]
                if Path(arg).is_absolute()
                and Path(arg).suffix.lower() in {".py", ".sh", ".pl", ".sif"}
                and Path(arg).is_file()
            )
        files.add(self.database)
        for helper in ("diamond", "prodigal", "nhmmer", "bedtools", "cmsearch", "cmscan"):
            if path := shutil.which(helper, path=environment.get("PATH", os.defpath)):
                files.add(Path(path).absolute())
        self._stamps = {path: _stamp(path) for path in sorted(files)}
        hashes = {str(path): sha(path) for path in self._stamps}
        self._environment = environment
        versions = {}
        for tool, (flag, expected) in TOOLS.items():
            probe = self.call(tool, [flag], work, threads=1, capture=True)
            if re.search(rf"(?<![\d.]){re.escape(expected)}(?![\d.])", probe) is None:
                raise ValueError(
                    f"Assessment requires {tool} {expected}. Version probe reported {probe.strip()!r}"
                )
            versions[tool] = expected
        self.verify_runtime()
        return {
            "policy": POLICY,
            "commands": self.commands,
            "runtime_files_sha256": hashes,
            "environment_sha256": {key: fingerprint(value) for key, value in self._environment.items()},
            "database": str(self.database),
            "database_sha256": hashes[str(self.database)],
            "versions": versions,
            "checkm2_threads": self.checkm2_threads,
            "rna_threads": self.rna_threads,
        }

    def verify_runtime(self) -> None:
        for path, stamp in self._stamps.items():
            if _stamp(path) != stamp:
                raise ValueError(f"Assessment runtime changed during execution: {path}")

    def validate_paths(self, output: Path) -> None:
        # The pinned tools interpolate some of these paths in their own shell commands.
        pattern = r"[A-Za-z0-9_./:\\+-]+" if os.name == "nt" else r"[A-Za-z0-9_./+-]+"
        for label, path in (("assessment output", output), ("checkm2_database", self.database)):
            if not re.fullmatch(pattern, str(path)) or not re.fullmatch(pattern, str(path.resolve())):
                raise ValueError(f"{label} needs a shell-safe path for the pinned quality tools: {path}")

    def call(
        self,
        tool: str,
        args: list[str],
        work: Path,
        *,
        threads: int,
        capture: bool = False,
        stdout: Path | None = None,
        stderr: Path | None = None,
    ) -> str:
        temp = work / "tmp"
        if self.paused():
            raise Paused
        self.verify_runtime()
        temp.mkdir(exist_ok=True)
        env = {
            **self._environment,
            "TMPDIR": str(temp),
            "TEMP": str(temp),
            "TMP": str(temp),
            "XDG_CACHE_HOME": str(temp / "cache"),
            "XDG_CONFIG_HOME": str(temp / "config"),
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "TF_NUM_INTRAOP_THREADS": str(threads),
            "TF_NUM_INTEROP_THREADS": "1",
        }
        command = [*self.commands[tool], *args]
        if capture:
            try:
                probe = subprocess.run(
                    command,
                    cwd=work,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdin=subprocess.DEVNULL,
                    timeout=60,
                )
            except subprocess.TimeoutExpired as error:
                raise ValueError(
                    f"{tool} version probe timed out after 60 seconds. Check {command[0]}"
                ) from error
            if probe.returncode:
                raise ValueError(f"{tool} version probe failed: {probe.stdout.strip()}")
            return probe.stdout
        assert stdout is not None and stderr is not None
        self.progress(tool, stderr)
        with stdout.open("wb") as out, stderr.open("wb") as err:
            with subprocess.Popen(
                command,
                cwd=work,
                env=env,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
                pass_fds=(self.lock_fd,) if os.name == "posix" and self.lock_fd is not None else (),
            ) as child:
                try:
                    while True:
                        try:
                            code = child.wait(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            if self.terminating():
                                raise Paused
                except (Paused, KeyboardInterrupt):
                    _stop_child(child)
                    raise
        if self.terminating():
            raise Paused
        self.verify_runtime()
        if code:
            raise ToolError(tool, code, stderr)
        return ""

    def checkm2(self, inputs: Path, attempt: Path, names: set[str]) -> dict[str, dict[str, Any]]:
        output = attempt / "results"
        try:
            self.call(
                "checkm2",
                [
                    "predict",
                    "--threads",
                    str(self.checkm2_threads),
                    "--lowmem",
                    "--extension",
                    ".fa",
                    "--database_path",
                    str(self.database),
                    "--input",
                    str(inputs),
                    "--output-directory",
                    str(output),
                    "--remove_intermediates",
                ],
                attempt,
                threads=self.checkm2_threads,
                stdout=attempt / "stdout.log",
                stderr=attempt / "stderr.log",
            )
        except ToolError as error:
            if error.code != 1 or (output / "quality_report.tsv").exists():
                raise
            try:
                return read_checkm2_attempt(attempt, names)
            except (OSError, ValueError):
                raise error from None
        return read_checkm2(output, names)

    def rna(self, fasta: Path, attempt: Path) -> dict[str, Any]:
        for domain, flag in (("bac", "-B"), ("arc", "-A")):
            self.call(
                "barrnap",
                [
                    "--kingdom",
                    domain,
                    "--threads",
                    str(self.rna_threads),
                    "--evalue",
                    "1e-6",
                    "--lencutoff",
                    "0.8",
                    "--reject",
                    "0.25",
                    str(fasta),
                ],
                attempt,
                threads=self.rna_threads,
                stdout=attempt / f"{domain}.rrna.gff",
                stderr=attempt / f"{domain}.barrnap.log",
            )
            self.call(
                "trnascan",
                [
                    flag,
                    "--thread",
                    str(self.rna_threads),
                    "-o",
                    str(attempt / f"{domain}.trna.tsv"),
                    str(fasta),
                ],
                attempt,
                threads=self.rna_threads,
                stdout=attempt / f"{domain}.trnascan.stdout.log",
                stderr=attempt / f"{domain}.trnascan.stderr.log",
            )
        return read_rna(attempt)


def _stop_child(child: subprocess.Popen[bytes]) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(child.pid, signal.SIGTERM)
        elif child.poll() is None:
            child.terminate()
        child.wait(timeout=3)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            if sys.platform != "win32":
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            pass
        child.wait()
