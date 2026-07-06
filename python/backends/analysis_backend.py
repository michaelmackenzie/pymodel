"""
Abstract interface for backend-specific statistical operations.

All statistical algorithms that are backend-agnostic (CLs scan,
Feldman-Cousins, NLL profile scan, …) are implemented in
``backends/analysis_common.py`` and call only the methods defined here.
Each concrete fitting backend (zfit, pyhf, …) provides a subclass that
implements these primitives using its own library calls.

The *state* object carried through every call is an opaque, backend-specific
context object that bundles the model, current data, minimiser, loss function,
and any other mutable state the backend needs.  The common algorithms never
inspect it directly; they only pass it back to the backend methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from backends.analysis_types import (
    CLsResult,
    FCResult,
    FitResult,
    HypothesisTestResult,
    NLLScanResult,
)


class AnalysisBackend(ABC):
    """Backend-specific statistical primitives consumed by the common algorithms."""

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, state: Any) -> FitResult:
        """Run an unconstrained MLE fit and return the result.

        The backend must update *state* so that subsequent calls to
        ``get_poi_value`` / ``parameter_values`` reflect the best-fit point.
        """

    @abstractmethod
    def fixed_poi_fit(self, state: Any, poi_value: float) -> FitResult:
        """Run an MLE fit with the POI fixed at *poi_value*.

        All nuisance parameters are left free to float.  The backend must
        restore the POI to its previous floating / fixed status after the call.
        """

    @abstractmethod
    def evaluate_nll(self, state: Any) -> float:
        """Return the NLL evaluated at the current parameter values."""

    # ------------------------------------------------------------------
    # Hypothesis testing
    # ------------------------------------------------------------------

    @abstractmethod
    def hypothesis_test(
        self,
        state: Any,
        poi_test: float,
        poi_alt: float = 0.0,
    ) -> HypothesisTestResult:
        """Compute observed CLs and expected CLs bands at a single POI value.

        Parameters
        ----------
        poi_test:
            The null-hypothesis POI value (the value being tested).
        poi_alt:
            The alternative-hypothesis POI value (typically 0 for upper limits).

        Returns
        -------
        HypothesisTestResult
            ``observed_cls`` is the ratio p_null / p_alt.
            ``expected_cls`` maps sigma integers {-2,-1,0,+1,+2} to the
            corresponding expected CLs values under the background-only
            hypothesis.
        """

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    @abstractmethod
    def get_poi_value(self, state: Any) -> float:
        """Return the current value of the parameter of interest."""

    @abstractmethod
    def set_poi_value(self, state: Any, value: float) -> None:
        """Set the parameter of interest to *value* (does not run a fit)."""

    @abstractmethod
    def poi_name(self, state: Any) -> str:
        """Return the name of the parameter of interest."""

    @abstractmethod
    def parameter_names(self, state: Any) -> List[str]:
        """Return the names of all floating parameters."""

    @abstractmethod
    def parameter_values(self, state: Any) -> Dict[str, float]:
        """Return a {name: value} dict for all floating parameters."""

    # ------------------------------------------------------------------
    # Parameter snapshots (for restoring state between iterations)
    # ------------------------------------------------------------------

    @abstractmethod
    def snapshot_parameters(self, state: Any) -> Any:
        """Capture a snapshot of the current parameter state.

        Returns an opaque object that can be passed to
        ``restore_parameters`` to reset the state.
        """

    @abstractmethod
    def restore_parameters(self, state: Any, snapshot: Any) -> None:
        """Restore parameter state from a previously captured snapshot."""

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_toy_data(self, state: Any) -> Any:
        """Draw a pseudo-dataset from the model at its current parameter values.

        Returns an opaque data object suitable for ``set_data``.
        """

    @abstractmethod
    def generate_asimov_data(self, state: Any) -> Any:
        """Return an Asimov (expected) dataset at current parameter values."""

    @abstractmethod
    def get_observed_data(self, state: Any) -> Any:
        """Return the observed (real) dataset from the fit model."""

    @abstractmethod
    def get_current_data(self, state: Any) -> Any:
        """Return the dataset currently loaded into *state*.

        This is the data that will be used by the next call to ``fit`` or
        ``evaluate_nll``.  For backends where the data is embedded in the loss
        function (e.g. zfit), this returns the dataset that was last passed to
        ``set_data``.
        """

    @abstractmethod
    def set_data(self, state: Any, data: Any) -> None:
        """Replace the current dataset in *state* with *data*.

        The backend must update the loss / likelihood internally so that
        subsequent fit / evaluate_nll calls use the new dataset.
        """

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    @abstractmethod
    def poi_uncertainty_hesse(self, state: Any, fit_result: FitResult) -> Optional[float]:
        """Return a Hessian-based POI uncertainty from a completed fit result.

        Returns None when the Hessian is unavailable or numerically unreliable.
        """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def signal_nominal_yield(self) -> Optional[float]:
        """Nominal signal yield for converting mu limits to yield limits, or None."""

    @property
    @abstractmethod
    def poi_is_signal_strength(self) -> bool:
        """True when the POI is a signal-strength modifier (mu), False otherwise."""

    @property
    def delta_nll_one_sigma(self) -> float:
        """Value of delta-NLL corresponding to the 1-sigma (68% CL) boundary.

        For backends that store NLL = -log L this is 0.5.
        For backends that store twice_nll = -2 log L this is 1.0.
        Used by ``estimate_poi_unc_from_profile`` to find the correct crossing.

        Subclasses should override when they store twice_nll in FitResult.nll.
        """
        return 0.5
