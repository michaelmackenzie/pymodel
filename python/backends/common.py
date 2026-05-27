import importlib


def package_version(module_name, fallback="unknown"):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return fallback
    return getattr(module, "__version__", fallback)


def add_feldman_cousins_arguments(parser):
    parser.add_argument(
        "--feldman-cousins", "-fc",
        type=float,
        default=None,
        metavar="ALPHA",
        help="If set, compute Feldman-Cousins confidence intervals using this alpha value (e.g. 0.1 for 90%% CL)",
    )
    parser.add_argument(
        "--fc-scan-points",
        type=int,
        default=21,
        help="Number of POI grid points for Feldman-Cousins construction",
    )
    parser.add_argument(
        "--fc-toys",
        type=int,
        default=100,
        help="Number of toys per POI grid point for Feldman-Cousins construction",
    )
    parser.add_argument(
        "--fc-scan-max",
        type=float,
        default=None,
        help="Optional POI scan maximum for Feldman-Cousins (default: automatic)",
    )


def add_source_arguments(parser, model_file_help):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-file", "-m", type=str, help=model_file_help)
    source.add_argument("--input-card", "-c", type=str, help="Path to a text model card")


def add_shared_analysis_arguments(
    parser,
    *,
    output_flag,
    output_default,
    output_help,
    poi_scan_max_help,
    include_plot_alias=False,
):
    parser.add_argument(
        "--toys", "-t",
        type=int,
        default=None,
        help="Number of toy datasets to generate and fit (default: use observed data if available, otherwise 1; use -1 for binned Asimov data)",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel worker processes for toy generation/fits (default: 1, no parallel processing)",
    )
    parser.add_argument("--cls", type=float, default=None, help="If set, compute the CLs upper limit for this alpha value")
    parser.add_argument("--signal-strength", type=float, default=None, help="Override the signal strength before generating toys")
    parser.add_argument(
        "--cls-scan-points",
        type=int,
        default=None,
        help="Number of scan points for CLs (default: 9 for counting+binned, else 25)",
    )
    parser.add_argument(
        "--cls-smart-scan",
        action="store_true",
        help="Adaptively refine the CLs scan range and granularity around the limit",
    )
    parser.add_argument(
        "--binned-bins",
        type=int,
        default=40,
        help="Number of bins for non-counting binned fits",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Save checkpoint every N datasets for progress tracking and recovery (default: no checkpoints)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume analysis from a previous checkpoint file (loads completed datasets and continues from index N+1)",
    )
    parser.add_argument(
        "--poi-scan-max",
        type=float,
        default=None,
        help=poi_scan_max_help,
    )

    if include_plot_alias:
        parser.add_argument(
            "--plot",
            "--plots",
            action="store_true",
            help="Save histogram plots of fit parameters, POI pull, and per-toy fit distributions",
        )
    else:
        parser.add_argument(
            "--plot",
            action="store_true",
            help="Save histogram plots of fit parameters, POI pull, and per-toy fit distributions",
        )

    parser.add_argument(
        "--nll-scan-points",
        type=int,
        default=121,
        help="Number of points in the profile-NLL scan used for --plot (default: 121)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots",
        help="Output directory for --plot artifacts",
    )
    parser.add_argument(
        "--ntoys-plot",
        "--ntoy-plots",
        dest="ntoys_plot",
        type=int,
        default=1,
        help="Number of datasets to individually plot when --plot is set (default: 1)",
    )
    parser.add_argument(
        "--set-parameters",
        type=str,
        default=None,
        help="Optional parameter settings to override the input model, e.g. 'param1=1.5,param2=2.0'",
    )
    parser.add_argument(
        "--freeze-parameters",
        type=str,
        default=None,
        help="Optional parameter list to freeze in the model, e.g. 'param1,param2'",
    )
    parser.add_argument(
        "--set-parameter-ranges",
        type=str,
        default=None,
        help="Optional parameter ranges to override the input model, e.g. 'param1=0.5:2.0,param2=1.0:3.0'",
    )
    parser.add_argument(
        output_flag,
        type=str,
        default=output_default,
        help=output_help,
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help=f"Optional JSON file path for ensemble evaluation report (default: derived from {output_flag})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible toy generation and minimization settings",
    )
