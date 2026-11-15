"""Command-line boundaries for generic preparation, proposals and external approvals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import read
from .data import Dataset, prepare_dataset
from .engine import ModelArtifact, ProposalArtifact, fit, propose, run
from .export import export
from .methods import METHOD_CHOICES, method_catalog
from .reconcile import SelectionArtifact, reconcile
from .resources import ResourceLimits


def _ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    """Report expected user-input errors concisely; --debug preserves tracebacks."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    debug = "--debug" in arguments
    arguments = [value for value in arguments if value != "--debug"]
    try:
        return _main(arguments)
    except (ValueError, OSError, ImportError, MemoryError) as error:
        if debug:
            raise
        print(f"bincomplete: error: {error}", file=sys.stderr)
        hint = "Use --debug for a traceback."
        if arguments[:1] not in (["encode"], ["cache"], ["assess"], ["report"]):
            hint += " Use preflight --config FILE to check experiment inputs."
        print(hint, file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    """Run one explicit package operation and print its artifact path.

    Parameters
    ----------
    argv : list of str, optional
        Arguments excluding the executable name; default uses ``sys.argv``.

    Returns
    -------
    int
        Zero on success; invalid inputs are reported by the argument parser.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="bincomplete",
        description="BinComplete: membership-guided contig recruitment; evaluation tools stay external.",
    )
    parser.add_argument("--version", action="version", version=f"BinComplete {__version__}")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for input/setup errors")
    commands = parser.add_subparsers(dest="command", required=True)
    catalogue = commands.add_parser(
        "methods", help="List the six algorithms, accepted names and stored profile IDs"
    )
    catalogue.add_argument("--json", action="store_true", help="Write a machine-readable method catalogue")
    initializer = commands.add_parser("init", help="Create the standard portable six-method experiment")
    initializer.add_argument("--output", type=Path, required=True)
    initializer.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cuda:0")
    initializer.add_argument("--embedding-mode", choices=["supplied", "encode"], default="supplied")
    initializer.add_argument(
        "--compact", action="store_true", help="Inherit shared settings from global config"
    )
    initializer.add_argument("--input-root", type=Path, help="Original mmlong2 output; implies --compact")
    initializer.add_argument("--global-config", type=Path)
    initializer.add_argument("--recipients", type=Path)
    for name in (
        "assembly",
        "bins",
        "memberships",
        "coverage",
        "groups",
        "source-ids",
        "donor-ids",
        "composition",
        "embeddings",
        "embedding-ids",
        "embedding-space",
    ):
        initializer.add_argument("--" + name, type=Path)
    initializer.add_argument("--coverage-profile", help="Exact JGI mean-column header")
    initializer.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        help="Subset of the standard methods; see bincomplete methods",
    )
    configuration = commands.add_parser("config", help="Create or inspect shared installation settings")
    config_commands = configuration.add_subparsers(dest="config_command", required=True)
    configure = config_commands.add_parser("init", help="Write an editable global configuration once")
    configure.add_argument("--output", type=Path)
    configure.add_argument("--encoder-path", type=Path)
    configure.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cpu")
    show = config_commands.add_parser(
        "show", help="Show global defaults and optional resolved method recipes"
    )
    show.add_argument("--path", type=Path)
    show.add_argument("--resolved", action="store_true")
    models = commands.add_parser("models", help="Manage local frozen encoder assets")
    model_commands = models.add_subparsers(dest="model_command", required=True)
    install = model_commands.add_parser("install", help="Download the pinned official DNABERT-S checkpoint")
    install.add_argument("model", choices=["dnabert_s"])
    install.add_argument("--output", type=Path)
    from .defaults import DNABERT_REVISION

    install.add_argument("--revision", default=DNABERT_REVISION)
    install.add_argument("--global-config", type=Path)
    inspect = model_commands.add_parser(
        "inspect", help="Describe local encoder-space metadata without loading or encoding"
    )
    inspect.add_argument("model", choices=["dnabert_s"])
    inspect.add_argument("--path", type=Path, required=True)
    inspect.add_argument("--encoder-options", type=Path)
    inspect.add_argument("--output", type=Path)
    encoder = commands.add_parser("encode", help="Precompute and save portable DNABERT-S embeddings")
    encoder.add_argument("--assembly", type=Path, required=True)
    encoder.add_argument("--output", type=Path, required=True, help="Directory for the embedding bundle")
    encoder.add_argument("--encoder-path", type=Path, help="Local DNABERT-S weights and tokenizer")
    encoder.add_argument(
        "--encoder-options", type=Path, help="JSON overrides for the configured encoder options"
    )
    encoder.add_argument("--global-config", type=Path)
    encoder.add_argument("--device", help="CPU or CUDA device; defaults to the configured device")
    encoder.add_argument("--cache", type=Path)
    encoder.add_argument("--threads", type=int)
    encoder.add_argument("--memory-gib", type=float)
    encoder.add_argument("--cache-gib", type=float)
    cache_parser = commands.add_parser("cache", help="Manage rebuildable BinComplete arrays")
    cache_commands = cache_parser.add_subparsers(dest="cache_command", required=True)
    cleaner = cache_commands.add_parser("clean", help="Preview or remove unused rebuildable cache entries")
    cleaner.add_argument("--cache", type=Path, help="Cache directory; defaults to the configured cache")
    cleaner.add_argument("--global-config", type=Path)
    cleaner.add_argument("--apply", action="store_true", help="Remove the selected entries")
    cleaner.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also discard inactive, recognized resumable entries",
    )
    doctor = commands.add_parser("doctor", help="Record dependencies and actual numerical device")
    doctor.add_argument("--device", default="cpu")
    doctor.add_argument("--output", type=Path)
    experiment = commands.add_parser(
        "experiment", help="Run explicitly selected methods from one TOML/JSON recipe"
    )
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--output", type=Path)
    check = commands.add_parser(
        "preflight", help="Check inputs and readiness without creating a run or fitting"
    )
    check.add_argument("--config", type=Path, required=True)
    demonstration = commands.add_parser("demo", help="Run a tiny six-method synthetic installation check")
    demonstration.add_argument("--output", type=Path, required=True)
    inventory = commands.add_parser(
        "candidates", help="Write a TSV inventory of saved proposals for external review"
    )
    inventory.add_argument("--proposals", type=Path, action="append", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument(
        "--priority-limit",
        type=int,
        default=40,
        help="Mark the first N proposals per method artifact/recipient as tier 1; retain every proposal",
    )
    inspector_report = commands.add_parser("report", help="Package existing results for BinCompleteInspector")
    inspector_report.add_argument("--config", type=Path, required=True)
    inspector_report.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Index generic or mmlong2 inputs in place")
    prepare.add_argument("--input-format", choices=["generic", "mmlong2"], default="generic")
    prepare.add_argument("--input-root", type=Path)
    for name in (
        "assembly",
        "assembly-index",
        "bins",
        "memberships",
        "coverage",
        "composition",
        "groups",
        "source-ids",
        "embeddings",
        "embedding-ids",
        "embedding-space",
        "encoder-path",
        "donor-ids",
    ):
        prepare.add_argument("--" + name, type=Path)
    prepare.add_argument("--coverage-profile")
    prepare.add_argument("--encoder-options", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("fit", help="Fit one explicit membership method")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--method", choices=METHOD_CHOICES, required=True)
    training.add_argument("--settings", type=Path)
    training.add_argument("--output", type=Path, required=True)
    proposal = commands.add_parser("propose", help="Use a completed model without fitting")
    proposal.add_argument("--dataset", type=Path, required=True)
    proposal.add_argument("--model", type=Path, required=True)
    proposal.add_argument("--recipients", type=Path)
    proposal.add_argument("--output", type=Path, required=True)
    sequence = commands.add_parser("run", help="Fit then propose one method")
    sequence.add_argument("--training", type=Path, required=True)
    sequence.add_argument("--dataset", type=Path, required=True)
    sequence.add_argument("--method", choices=METHOD_CHOICES, required=True)
    sequence.add_argument("--settings", type=Path)
    sequence.add_argument("--output", type=Path, required=True)
    reconcile_command = commands.add_parser(
        "reconcile", help="Select structurally compatible externally approved IDs"
    )
    reconcile_command.add_argument("--proposals", type=Path, action="append", required=True)
    reconcile_command.add_argument("--approved-ids", type=Path, required=True)
    reconcile_command.add_argument("--protected-bins", type=Path)
    reconcile_command.add_argument("--time-limit", type=float, default=60.0)
    reconcile_command.add_argument("--output", type=Path, required=True)
    assessment = commands.add_parser("assess", help="Assess saved proposals with CheckM2 and RNA tools")
    assessment.add_argument("--experiment", type=Path, required=True)
    assessment.add_argument("--config", type=Path, required=True)
    assessment.add_argument("--output", type=Path, required=True)
    assessment.add_argument("--mode", choices=("greedy", "full"), required=True)
    assessment.add_argument("--max-rounds", type=int)
    assessment.add_argument("--reuse-from", type=Path)
    assessment.add_argument("--memory-gib", type=float, default=4.0)
    exporter = commands.add_parser("export", help="Materialize selected FASTAs once")
    source = exporter.add_mutually_exclusive_group(required=True)
    source.add_argument("--selection", type=Path)
    source.add_argument("--proposals", type=Path)
    exporter.add_argument("--proposal-ids", type=Path)
    exporter.add_argument("--complete-directory", action="store_true")
    exporter.add_argument("--output", type=Path, required=True)
    for command in (prepare, training, proposal, sequence, reconcile_command, exporter, inventory):
        command.add_argument("--memory-gib", type=float, default=4.0)
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--cache-gib", type=float, default=10.0)
        command.add_argument("--chunk-size", type=int, default=4096)
        if command in (training, proposal, sequence):
            command.add_argument("--cache", type=Path)
    args = parser.parse_args(arguments)
    if args.command == "assess":
        from .assessment import assess
        from .assessment_tools import Paused

        try:
            assessment_summary = assess(
                args.experiment,
                args.config,
                args.output,
                mode=args.mode,
                max_rounds=args.max_rounds,
                reuse_from=args.reuse_from,
                memory_gib=args.memory_gib,
            )
        except Paused:
            print("Assessment paused during tool validation. Rerun to resume.", file=sys.stderr)
            return 75
        print(assessment_summary)
        return 75 if read(assessment_summary)["state"] == "PAUSED" else 0
    if args.command == "encode":
        from .defaults import load_defaults
        from .encoding import encode

        defaults, _ = load_defaults(args.global_config)
        options = dict(defaults["encoder"], device=defaults["device"])
        if args.encoder_options is not None:
            overrides = read(args.encoder_options)
            if not isinstance(overrides, dict):
                raise ValueError("Encoder options must be a JSON object")
            options.update(overrides)
        if args.device is not None:
            options["device"] = args.device
        if options["precision"] == "auto":
            options["precision"] = "float32" if options["device"] == "cpu" else "bfloat16"
        resources = dict(defaults["resources"])
        for name in ("threads", "memory_gib", "cache_gib"):
            if getattr(args, name) is not None:
                resources[name] = getattr(args, name)
        print(
            encode(
                args.assembly,
                args.output,
                encoder_path=args.encoder_path or Path(defaults["encoder_path"]),
                encoder_options=options,
                cache_root=args.cache or Path(defaults["cache"]),
                limits=ResourceLimits(**resources),
            )
        )
        return 0
    if args.command == "cache":
        from .cache import DerivedCache
        from .defaults import load_defaults

        root = args.cache
        if root is None:
            defaults, _ = load_defaults(args.global_config)
            root = Path(defaults["cache"])
        cache = DerivedCache(root.expanduser().resolve(), ResourceLimits())
        print(json.dumps(cache.clean(apply=args.apply, include_incomplete=args.include_incomplete), indent=2))
        return 0
    if args.command == "methods":
        catalogue_rows = method_catalog()
        if args.json:
            print(json.dumps(catalogue_rows, indent=2))
        else:
            for row in catalogue_rows:
                print(f"{row['name']} [profile: {row['profile_id']}]: {row['description']}")
        return 0
    if args.command == "demo":
        from .demo import demo

        print(demo(args.output))
        return 0
    if args.command == "candidates":
        from .inventory import candidate_table

        print(
            candidate_table(
                [ProposalArtifact(path) for path in args.proposals],
                args.output,
                limits=ResourceLimits(args.memory_gib, args.threads, args.cache_gib, args.chunk_size),
                priority_limit=args.priority_limit,
            )
        )
        return 0
    if args.command == "report":
        from .report import write_report

        print(write_report(args.config, args.output))
        return 0
    if args.command == "config":
        from .defaults import load_defaults, method_defaults, write_defaults

        if args.config_command == "init":
            print(write_defaults(args.output, encoder_path=args.encoder_path, device=args.device))
        else:
            values, provenance = load_defaults(args.path)
            view = {"configuration": values, "provenance": provenance}
            if args.resolved:
                from .core import build
                from .methods import recipe

                limits = ResourceLimits(**values["resources"])
                settings = method_defaults(values)
                view["method_recipes"] = {
                    method: recipe(method, limits, settings[method]) for method in values["enabled_methods"]
                }
                for spec in view["method_recipes"].values():
                    for category in (
                        "retriever",
                        "masker",
                        "negative_sampler",
                        "scorer",
                        "search",
                        "classifier",
                    ):
                        if spec[category] is not None:
                            build(category, spec[category])
            print(json.dumps(view, indent=2))
        return 0
    if args.command == "models":
        if args.model_command == "inspect":
            from .artifacts import freeze
            from .representations.dna import local_contract

            space, _ = local_contract(args.path, read(args.encoder_options) if args.encoder_options else None)
            if args.output is not None:
                freeze(args.output, space)
            print(json.dumps(space, indent=2))
            return 0
        from .model_assets import install_dnabert

        installation = install_dnabert(args.output, revision=args.revision, global_config=args.global_config)
        print(json.dumps(installation, indent=2))
        return 0
    if args.command == "init":
        from .standard import initialize, initialize_compact

        inputs = {
            name: getattr(args, name)
            for name in (
                "assembly",
                "bins",
                "memberships",
                "coverage",
                "coverage_profile",
                "groups",
                "source_ids",
                "donor_ids",
                "composition",
                "embeddings",
                "embedding_ids",
                "embedding_space",
            )
            if getattr(args, name) is not None
        }
        if args.compact or args.input_root is not None or inputs or args.methods is not None:
            if any(x == "--embedding-mode" or x.startswith("--embedding-mode=") for x in arguments):
                parser.error(
                    "Compact experiments inherit encoder settings; declare supplied vectors in their dataset"
                )
            # An omitted --device inherits the global device instead of the
            # standalone initializer's historical CUDA default.
            explicit_device = (
                args.device if any(x == "--device" or x.startswith("--device=") for x in arguments) else None
            )
            print(
                initialize_compact(
                    args.output,
                    input_root=args.input_root,
                    global_config=args.global_config,
                    recipients=args.recipients,
                    device=explicit_device,
                    inputs=inputs,
                    methods=args.methods,
                )
            )
        else:
            if args.global_config is not None or args.recipients is not None:
                parser.error("--global-config and --recipients require --compact or --input-root")
            print(initialize(args.output, device=args.device, embedding_mode=args.embedding_mode))
        return 0
    if args.command == "preflight":
        from .preflight import preflight

        report = preflight(args.config)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2
    if args.command == "doctor":
        from .environment import environment_report

        report = environment_report(args.device)
        content = json.dumps(report, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
        print(content, end="")
        return 1 if report["issues"] else 0
    if args.command == "experiment":
        from .experiments import run_experiment

        experiment_result = run_experiment(
            args.config, output=args.output, progress=lambda row: print(json.dumps(row), flush=True)
        )
        print(str(experiment_result.path))
        return 0 if experiment_result.manifest["status"] == "SUCCEEDED" else 1
    result: Dataset | ModelArtifact | ProposalArtifact | SelectionArtifact | Path
    limits = ResourceLimits(args.memory_gib, args.threads, args.cache_gib, args.chunk_size)
    if args.command == "prepare":
        names = (
            "assembly",
            "assembly_index",
            "bins",
            "memberships",
            "coverage",
            "composition",
            "coverage_profile",
            "groups",
            "source_ids",
            "embeddings",
            "embedding_ids",
            "embedding_space",
            "encoder_path",
            "donor_ids",
            "input_format",
            "input_root",
        )
        result = prepare_dataset(
            args.output,
            limits=limits,
            encoder_options=read(args.encoder_options) if args.encoder_options else None,
            **{name: getattr(args, name) for name in names},
        )
    elif args.command == "fit":
        result = fit(
            Dataset(args.dataset),
            args.output,
            method=args.method,
            settings=read(args.settings) if args.settings else None,
            limits=limits,
            cache=args.cache,
        )
    elif args.command == "propose":
        result = propose(
            ModelArtifact(args.model),
            Dataset(args.dataset),
            args.output,
            recipients=_ids(args.recipients),
            limits=limits,
            cache=args.cache,
        )
        overlap = result.manifest["overlap"]
        print(
            json.dumps(
                {"overlap_mode": overlap["mode"], "overlapping_recipients": overlap["overlapping_recipients"]}
            )
        )
    elif args.command == "run":
        result = run(
            Dataset(args.training),
            Dataset(args.dataset),
            args.output,
            method=args.method,
            settings=read(args.settings) if args.settings else None,
            limits=limits,
            cache=args.cache,
        )
        overlap = result.manifest["overlap"]
        print(
            json.dumps(
                {"overlap_mode": overlap["mode"], "overlapping_recipients": overlap["overlapping_recipients"]}
            )
        )
    elif args.command == "reconcile":
        result = reconcile(
            [ProposalArtifact(path) for path in args.proposals],
            _ids(args.approved_ids) or [],
            args.output,
            protected_bins=_ids(args.protected_bins),
            time_limit=args.time_limit,
            limits=limits,
        )
    else:
        artifact = SelectionArtifact(args.selection) if args.selection else ProposalArtifact(args.proposals)
        result = export(
            artifact,
            args.output,
            proposal_ids=_ids(args.proposal_ids),
            complete_directory=args.complete_directory,
            limits=limits,
        )
    print(str(result if isinstance(result, Path) else result.path))
    return 0
