"""
Common result types shared across all analysis backends.

These dataclasses define the contract between the backend-agnostic algorithms
(CLs scan, Feldman-Cousins, NLL profile scan, …) and the low-level
backend-specific adapters that evaluate likelihoods and manipulate parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level fit result
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Outcome of a single unconstrained or fixed-POI MLE fit."""

    valid: bool
    """True when the minimisation converged and the result is trustworthy."""

    poi_value: float
    """Best-fit value of the parameter of interest."""

    poi_uncertainty: Optional[float]
    """Hessian (symmetric) uncertainty on the POI, or None if unavailable."""

    nll: float
    """Value of the negative log-likelihood (or a monotone proxy) at the best fit.

    The scale convention is backend-specific:
    - zmodel (zfit): stores NLL = -log L
    - hfmodel (pyhf): stores twice_nll = -2 log L (returned by pyhf.infer.mle)

    Consumers of this field must treat it as an opaque value for *relative*
    comparisons within a single backend (e.g. delta-NLL for Feldman-Cousins),
    not as an absolute NLL suitable for cross-backend comparison.
    """

    param_values: Dict[str, float] = field(default_factory=dict)
    """All floating parameter values keyed by parameter name."""

    param_uncertainties: Dict[str, float] = field(default_factory=dict)
    """Hessian uncertainties for all floating parameters (best-effort)."""

    edm: Optional[float] = None
    """Estimated distance to minimum (backend-specific, may be None)."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Backend-specific auxiliary information (fit status objects, etc.)."""


# ---------------------------------------------------------------------------
# Hypothesis test result (one POI point)
# ---------------------------------------------------------------------------

@dataclass
class HypothesisTestResult:
    """CLs values returned by a single call to a hypothesis test at one POI."""

    observed_cls: float
    """p_s+b / p_b (or NaN if the test could not be evaluated)."""

    expected_cls: Dict[int, float] = field(default_factory=dict)
    """Expected CLs at Gaussian sigma offsets: keys are -2, -1, 0, +1, +2."""


# ---------------------------------------------------------------------------
# CLs scan result (full scan over a range of POI values)
# ---------------------------------------------------------------------------

@dataclass
class CLsResult:
    """Outcome of a full CLs scan over a grid of POI values."""

    observed_limit: Optional[float]
    """Observed CLs upper limit (POI where CLs_obs crosses alpha)."""

    expected_limit: Optional[float]
    """Median expected CLs upper limit (0-sigma band crossing)."""

    expected_quantiles: Dict[str, float] = field(default_factory=dict)
    """Expected-band quantiles: keys '2.5%', '16%', '50%', '84%', '97.5%'."""

    scan_points: int = 0
    """Number of POI grid points used in the scan."""

    scan_max: float = 0.0
    """Upper end of the scanned POI range."""

    curve: Optional[Dict[str, Any]] = None
    """Full scan curve for plotting: pois, observed_cls, expected_cls_bands."""


# ---------------------------------------------------------------------------
# Feldman-Cousins result
# ---------------------------------------------------------------------------

@dataclass
class FCResult:
    """Outcome of a Feldman-Cousins confidence-interval construction."""

    interval: Optional[Tuple[float, float]]
    """(lower, upper) bounds of the FC confidence interval, or None."""

    status: str = "ok"
    """Human-readable status string ('ok', 'no-accepted-points', 'failed', …)."""

    alpha: float = 0.05
    """Significance level used (1 - CL)."""

    poi_name: str = ""
    """Name of the parameter of interest."""

    grid: Dict[str, Any] = field(default_factory=dict)
    """Diagnostic information: poi grid, q_obs values, q_crit values, etc."""

    scan_points: int = 0
    n_toys: int = 0
    scan_max: float = 0.0


# ---------------------------------------------------------------------------
# NLL profile scan result
# ---------------------------------------------------------------------------

@dataclass
class NLLScanResult:
    """delta-NLL profile scan of the POI for plotting."""

    poi_name: str
    poi_values: List[float]
    delta_nll_values: List[float]
