import argparse
from typing import Dict

from backends import get_backends
from backends.base import BaseBackend


def _print_startup_header(backend):
    title = f"{backend.name} statistical analysis"
    version_string = " | ".join(f"{name} {version}" for name, version in backend.runtime_versions())
    width = max(len(title), len(version_string)) + 4
    print("*" * width, flush=True)
    print("*" + title.center(width - 2) + "*", flush=True)
    print("*" + version_string.center(width - 2) + "*", flush=True)
    print("*" * width, flush=True)


def _add_backend_command_parsers(backend_parser, backend):
    subparsers = backend_parser.add_subparsers(dest="command", required=False)

    build_parser = subparsers.add_parser("build", help="Build and save a model from a text card")
    backend.add_build_arguments(build_parser)

    load_parser = subparsers.add_parser("load", help="Load and summarize a saved model bundle")
    backend.add_load_arguments(load_parser)

    analyze_parser = subparsers.add_parser("analyze", help="Run toy fits and optional CLs limits")
    backend.add_analyze_arguments(analyze_parser)


def build_parser(backends: Dict[str, BaseBackend]):
    parser = argparse.ArgumentParser(description="Top-level CLI for pyhf/zfit/RooFit model workflows")
    backend_subparsers = parser.add_subparsers(dest="backend_name", required=True)

    for backend_name, backend in backends.items():
        backend_parser = backend_subparsers.add_parser(backend_name, help=backend.description)
        _add_backend_command_parsers(backend_parser, backend)

    return parser


def _print_load_summary(summary):
    print(f"Loaded model from {summary['model_path']}")
    print(f"Model name: {summary['model_name']}")
    print(f"Observable range: {summary['obs_range']}")
    if summary.get("channels"):
        print(f"Channels: {', '.join(summary['channels'])}")
    print(f"Processes: {summary['processes']}")
    print(f"Signal process: {summary['signal_process']}")
    if summary.get("poi_name") is not None:
        print(f"POI name: {summary['poi_name']}")
    print(f"Constraints: {summary['constraints']}")
    print(f"Floating params: {summary['floating_params']}")
    if summary.get("observed_count") is not None:
        print(f"Observed data count: {summary['observed_count']}")
    if summary.get("pdf_lines"):
        print("PDFs and variables:")
        for line in summary["pdf_lines"]:
            print(line)


def run(argv=None):
    backends = get_backends()
    parser = build_parser(backends)
    
    # If no subcommand (build/load/analyze) is provided, default to "analyze"
    # by injecting it into argv before parsing.
    if argv is None:
        import sys
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    
    # Structure: pymodel <backend> [subcommand] [args...]
    # argv[0] = backend name (e.g., 'roomodel', 'hfmodel', 'zmodel')
    # argv[1] = either a subcommand ('build', 'load', 'analyze') or an option/file
    # 
    # If argv has at least 2 elements and argv[1] is NOT a known subcommand
    # and does NOT start with '-', then it's likely a file/positional arg for
    # the default analyze command. Inject 'analyze'.
    known_subcommands = {"build", "load", "analyze"}
    
    if len(argv) > 1 and argv[1] not in known_subcommands and not argv[1].startswith("-"):
        # argv[1] looks like a file or positional arg, not a subcommand
        argv.insert(1, "analyze")
    elif len(argv) > 1 and argv[1].startswith("-"):
        # argv[1] is an option flag (e.g., '-c', '--input-card'), so we need 'analyze'
        argv.insert(1, "analyze")
    
    args = parser.parse_args(argv)

    backend = backends[args.backend_name]
    _print_startup_header(backend)

    command = getattr(args, "command", "analyze") or "analyze"

    if command == "build":
        output_path = backend.build_model(args.input_card, args.output_file)
        print(f"Saved FitModel to {output_path}")
        return 0

    if command == "load":
        summary = backend.load_summary(args.model_file, verbose=args.verbose)
        _print_load_summary(summary)
        return 0

    if command == "analyze":
        backend.run_analysis(args)
        return 0

    parser.error(f"Unknown command: {command}")
    return 2
