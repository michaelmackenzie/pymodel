from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """Common interface for pymodel backends."""

    name = ""
    description = ""

    @abstractmethod
    def runtime_versions(self):
        """Return list of (name, version) tuples for startup header."""

    @abstractmethod
    def build_defaults(self):
        """Return build command defaults and help metadata."""

    def add_build_arguments(self, parser):
        defaults = self.build_defaults()
        parser.add_argument("input_card", help="Path to the model-card text file")
        parser.add_argument(
            "output_file",
            nargs="?",
            default=defaults["output_file"],
            help=defaults["output_help"],
        )

    def add_load_arguments(self, parser):
        parser.add_argument("model_file", help="Path to the saved model file")
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Increase load output detail; use -vv for PDF and variable listings",
        )

    @abstractmethod
    def add_analyze_arguments(self, parser):
        """Add backend-specific and shared analyze options."""

    @abstractmethod
    def build_model(self, input_card, output_file):
        """Build model bundle from input card and save to output path."""

    @abstractmethod
    def load_summary(self, model_file, verbose=0):
        """Load a saved model and return summary dict."""

    @abstractmethod
    def run_analysis(self, args):
        """Run backend analysis entrypoint for parsed analyze args."""
