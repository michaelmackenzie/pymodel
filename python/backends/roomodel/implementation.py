from backends.base import BaseBackend
from backends.common import (
    add_feldman_cousins_arguments,
    add_shared_analysis_arguments,
    add_source_arguments,
    package_version,
)


class RooModelBackend(BaseBackend):
    name = "roomodel"
    description = "Use RooFit-based fitting implementation"
    output_argument = "output"

    def runtime_versions(self):
        from roomodel import __version__ as roomodel_version

        return [
            ("roomodel", roomodel_version),
            ("ROOT", package_version("ROOT")),
        ]

    def add_analyze_arguments(self, parser):
        add_feldman_cousins_arguments(parser)
        add_source_arguments(parser, "Path to a saved model file (default format: .root)")
        parser.add_argument(
            "--fit-mode",
            choices=("auto", "binned", "unbinned"),
            default="auto",
            help="Likelihood type: auto (default), binned, or unbinned",
        )
        add_shared_analysis_arguments(
            parser,
            output_flag="--output",
            output_default="analysis_output_roomodel.json",
            output_help="Output analysis snapshot path (JSON is written)",
            poi_scan_max_help="Optional upper edge for POI scan (currently reserved for future CLs/profile scans)",
        )

    def build_defaults(self):
        return {
            "output_file": "model.root",
            "output_help": "Path to the output model file (default: model.root)",
        }

    def build_model(self, input_card, output_file):
        from roomodel.build_model_from_text import build_and_save_model_from_card_file

        return build_and_save_model_from_card_file(input_card, output_file)

    def load_summary(self, model_file, verbose=0):
        from roomodel.load_model import load_and_summarize_model

        return load_and_summarize_model(model_file, verbose=verbose)

    def run_analysis(self, args):
        from roomodel.analyze_model import run_analysis_cli

        run_analysis_cli(args)


def create_backend():
    return RooModelBackend()
