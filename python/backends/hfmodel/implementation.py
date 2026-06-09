from backends.common import (
    add_feldman_cousins_arguments,
    add_shared_analysis_arguments,
    add_source_arguments,
    package_version,
)
from backends.base import BaseBackend


class HFModelBackend(BaseBackend):
    name = "hfmodel"
    description = "Use pyhf-based fitting implementation"
    output_argument = "output"

    def runtime_versions(self):
        from hfmodel import __version__ as hfmodel_version

        return [
            ("hfmodel", hfmodel_version),
            ("pyhf", package_version("pyhf")),
        ]

    def add_analyze_arguments(self, parser):
        parser.add_argument(
            "--backend",
            choices=("scipy", "minuit", "jax"),
            default="scipy",
            help="pyhf optimizer backend: scipy (default), minuit, or jax",
        )
        parser.add_argument(
            "--hessian-method",
            choices=("auto", "manual", "minuit", "jax"),
            default="auto",
            help="Hessian extraction strategy: auto (backend-aware), manual finite-difference, minuit, or jax",
        )
        add_feldman_cousins_arguments(parser)
        add_source_arguments(parser, "Path to a saved model file (default format: .json)")
        add_shared_analysis_arguments(
            parser,
            output_flag="--output",
            output_default="analysis_output.json",
            output_help="Output analysis snapshot path (JSON is written)",
            poi_scan_max_help="Upper edge for POI scans (used by CLs and profile scans; defaults to POI upper bound)",
        )

    def build_defaults(self):
        return {
            "output_file": "model.json",
            "output_help": "Path to the output model file (default: model.json)",
        }

    def build_model(self, input_card, output_file):
        from hfmodel.build_model_from_text import build_and_save_model_from_card_file

        return build_and_save_model_from_card_file(input_card, output_file)

    def load_summary(self, model_file, verbose=0):
        from hfmodel.load_model import load_and_summarize_model

        return load_and_summarize_model(model_file, verbose=verbose)

    def run_analysis(self, args):
        from hfmodel.analyze_model import run_analysis_cli

        run_analysis_cli(args)


def create_backend():
    return HFModelBackend()
