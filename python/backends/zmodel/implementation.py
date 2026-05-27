import argparse

from backends.common import (
    add_feldman_cousins_arguments,
    add_shared_analysis_arguments,
    add_source_arguments,
    package_version,
)
from backends.base import BaseBackend


class ZModelBackend(BaseBackend):
    name = "zmodel"
    description = "Use zfit-based fitting implementation"
    output_argument = "output"

    def runtime_versions(self):
        from zmodel import __version__ as zmodel_version

        return [
            ("zmodel", zmodel_version),
            ("zfit", package_version("zfit")),
            ("hepstats", package_version("hepstats")),
        ]

    def add_analyze_arguments(self, parser):
        add_feldman_cousins_arguments(parser)
        add_source_arguments(parser, "Path to a saved model file (default format: .pkl)")
        parser.add_argument("--scan-max", type=float, default=None, help="Override the maximum signal scan value for CLs")
        parser.add_argument(
            "--fit-mode",
            choices=("auto", "binned", "unbinned"),
            default="auto",
            help="Likelihood type: auto (default), binned, or unbinned",
        )
        parser.add_argument(
            "--graph-mode",
            choices=("auto", "on", "off"),
            default="on",
            help="TensorFlow graph mode: on (default), auto, or off",
        )
        parser.add_argument(
            "--profile-scan",
            action="store_true",
            help="For counting+binned fits, run a profile-likelihood scan over the POI instead of full minimization",
        )
        parser.add_argument(
            "--poi-name",
            default=None,
            help="Optional parameter name to treat as parameter of interest (POI) for --profile-scan",
        )
        parser.add_argument(
            "--promote-poi",
            action="store_true",
            help="If --poi-name points to a fixed parameter, set it floating so it can be used as POI",
        )
        parser.add_argument(
            "--poi-scan-points",
            type=int,
            default=41,
            help="Number of scan points for --profile-scan",
        )
        add_shared_analysis_arguments(
            parser,
            output_flag="--output",
            output_default="analysis_output.pkl",
            output_help="Output pickle file containing the fitted model, input data, and toy summaries",
            poi_scan_max_help="Optional upper edge for POI scan (lower edge uses POI lower bound/default)",
        )
        parser.add_argument(
            "--output-pkl",
            dest="output",
            default=argparse.SUPPRESS,
            help="Compatibility alias for --output",
        )

    def build_defaults(self):
        return {
            "output_file": "model.pkl",
            "output_help": "Path to the output model file (default: model.pkl)",
        }

    def build_model(self, input_card, output_file):
        from zmodel.build_model_from_text import build_and_save_model_from_card_file

        return build_and_save_model_from_card_file(input_card, output_file)

    def load_summary(self, model_file, verbose=0):
        from zmodel.load_model import load_and_summarize_model

        return load_and_summarize_model(model_file, verbose=verbose)

    def run_analysis(self, args):
        from zmodel.analyze_model import run_analysis_cli

        run_analysis_cli(args)


def create_backend():
    return ZModelBackend()
