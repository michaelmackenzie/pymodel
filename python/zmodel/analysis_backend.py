"""
zfit / hepstats backend adapter for the common analysis algorithms.

This module implements ``AnalysisBackend`` using zfit as the minimiser and
``hepstats.hypotests.calculators.AsymptoticCalculator`` for hypothesis testing.

The *state* object used here is a ``ZfitAnalysisState`` dataclass that carries
everything needed to run a fit iteration: the fit model, the current dataset,
the loss function, and the minimiser.  The common algorithms in
``backends/analysis_common.py`` never inspect the state directly; they only
pass it back through the interface methods defined here.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import zfit

from backends.analysis_backend import AnalysisBackend
from backends.analysis_types import FitResult, HypothesisTestResult

from backends.zfit_parameter_utils import (
    all_params_list as _all_params,
    capture_fit_model_parameter_values as _capture_fit_model_params,
    restore_parameter_values as _restore_params,
    channel_models as _channel_models,
)
from zmodel.utilities import AsymptoticCalculator, POI


# ---------------------------------------------------------------------------
# State object
# ---------------------------------------------------------------------------

@dataclass
class ZfitAnalysisState:
    """Mutable context object bundling all zfit fit-iteration state."""

    fit_model: Any
    """The zmodel FitModel instance."""

    resolved_fit_mode: str
    """'binned' or 'unbinned'."""

    binned_model: Any
    """Binned PDF (or dict of per-channel binned PDFs) used in binned fits."""

    binned_space: Any
    """zfit.Space (or dict) defining the binning."""

    is_counting: bool
    """True when the model is a counting experiment."""

    minimizer: Any
    """zfit Minuit minimiser instance."""

    signal_param: Any
    """The signal-strength (or equivalent) zfit parameter."""

    poi_param: Any
    """The parameter of interest (may be the same as signal_param)."""

    # Current data and loss – updated each iteration via set_data()
    current_data: Any = None
    current_loss: Any = None

    # Metadata propagated from fit_model
    _signal_nominal_yield: Optional[float] = field(default=None, repr=False)
    _poi_is_signal_strength: bool = field(default=True, repr=False)


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------

class ZfitAnalysisBackend(AnalysisBackend):
    """AnalysisBackend implementation wrapping zfit and hepstats."""

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, state: ZfitAnalysisState) -> FitResult:
        if state.current_loss is None:
            raise RuntimeError("ZfitAnalysisState.current_loss is not set; call set_data() first")

        result = state.minimizer.minimize(state.current_loss)
        poi_unc = _extract_hesse_error(result, state.poi_param)

        return FitResult(
            valid=bool(result.valid),
            poi_value=float(state.poi_param.value()),
            poi_uncertainty=poi_unc,
            nll=float(state.current_loss.value()),
            param_values=_extract_param_values(result),
            param_uncertainties=_extract_param_hesse_errors(result),
            edm=float(result.edm) if result.edm is not None else None,
            extra={"zfit_valid": bool(result.valid)},
        )

    def fixed_poi_fit(self, state: ZfitAnalysisState, poi_value: float) -> FitResult:
        if state.current_loss is None:
            raise RuntimeError("ZfitAnalysisState.current_loss is not set")

        was_floating = bool(getattr(state.poi_param, "floating", True))
        try:
            state.poi_param.floating = False
            state.poi_param.set_value(float(poi_value))
            result = state.minimizer.minimize(state.current_loss)
            nll_val = float(state.current_loss.value())
            poi_unc = _extract_hesse_error(result, state.poi_param)
        finally:
            state.poi_param.floating = was_floating

        return FitResult(
            valid=bool(result.valid),
            poi_value=float(poi_value),
            poi_uncertainty=poi_unc,
            nll=nll_val,
            param_values=_extract_param_values(result),
            param_uncertainties={},
        )

    def evaluate_nll(self, state: ZfitAnalysisState) -> float:
        if state.current_loss is None:
            raise RuntimeError("ZfitAnalysisState.current_loss is not set")
        return float(state.current_loss.value())

    # ------------------------------------------------------------------
    # Hypothesis testing
    # ------------------------------------------------------------------

    def hypothesis_test(
        self,
        state: ZfitAnalysisState,
        poi_test: float,
        poi_alt: float = 0.0,
    ) -> HypothesisTestResult:
        if state.current_loss is None:
            raise RuntimeError("ZfitAnalysisState.current_loss is not set")

        calculator = AsymptoticCalculator(input=state.current_loss, minimizer=state.minimizer)

        poinull = POI(state.signal_param, float(poi_test))
        poialt = POI(state.signal_param, float(poi_alt))

        def _to_float(value):
            arr = np.asarray(value, dtype=float).reshape(-1)
            return float(arr[0]) if arr.size > 0 else float("nan")

        pnull, palt = calculator.pvalue(poinull, poialt)
        pnull = _to_float(pnull)
        palt = _to_float(palt)

        if palt <= 0.0 or not np.isfinite(palt):
            obs_cls = float("nan")
        else:
            obs_cls = pnull / palt

        expected_curves = calculator.expected_pvalue(
            poinull,
            poialt,
            nsigma=[-2, -1, 0, 1, 2],
            CLs=True,
        )
        expected_cls = {}
        for sigma, curve in zip((-2, -1, 0, 1, 2), expected_curves):
            expected_cls[sigma] = _to_float(curve)

        return HypothesisTestResult(
            observed_cls=obs_cls,
            expected_cls=expected_cls,
        )

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_poi_value(self, state: ZfitAnalysisState) -> float:
        return float(state.poi_param.value())

    def set_poi_value(self, state: ZfitAnalysisState, value: float) -> None:
        state.poi_param.set_value(float(value))

    def poi_name(self, state: ZfitAnalysisState) -> str:
        return str(state.poi_param.name)

    def parameter_names(self, state: ZfitAnalysisState) -> List[str]:
        return [p.name for p in _all_params(state.fit_model)]

    def parameter_values(self, state: ZfitAnalysisState) -> Dict[str, float]:
        return {p.name: float(p.value()) for p in _all_params(state.fit_model)}

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_parameters(self, state: ZfitAnalysisState) -> Dict:
        return _capture_fit_model_params(state.fit_model)

    def restore_parameters(self, state: ZfitAnalysisState, snapshot: Dict) -> None:
        _restore_params(snapshot)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_toy_data(self, state: ZfitAnalysisState) -> Any:
        """Generate a pseudo-dataset using the current model parameters."""
        # Import here to avoid circular imports; these helpers live in
        # analysis_core and are kept there until that file is thinned.
        from zmodel.analysis_core import _build_toy_data
        data, _count, _plot = _build_toy_data(
            fit_model=state.fit_model,
            resolved_fit_mode=state.resolved_fit_mode,
            binned_space=state.binned_space,
            is_counting=state.is_counting,
        )
        return data

    def generate_asimov_data(self, state: ZfitAnalysisState) -> Any:
        from zmodel.analysis_core import _build_asimov_binned_data
        data, _counts, _plot = _build_asimov_binned_data(
            state.binned_model, state.binned_space, state.fit_model
        )
        return data

    def get_observed_data(self, state: ZfitAnalysisState) -> Any:
        return state.fit_model.data

    def set_data(self, state: ZfitAnalysisState, data: Any) -> None:
        """Replace the current dataset and rebuild the loss function."""
        state.current_data = data
        state.current_loss = _build_loss(
            fit_model=state.fit_model,
            resolved_fit_mode=state.resolved_fit_mode,
            binned_model=state.binned_model,
            data=data,
        )

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def poi_uncertainty_hesse(
        self, state: ZfitAnalysisState, fit_result: FitResult
    ) -> Optional[float]:
        # The uncertainty was already extracted during fit(); return it.
        return fit_result.poi_uncertainty

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def signal_nominal_yield(self) -> Optional[float]:
        # Accessed via state in practice; this property is on the adapter.
        return None  # Override when constructing via ZfitAnalysisState._signal_nominal_yield

    @property
    def poi_is_signal_strength(self) -> bool:
        return True  # Override when constructing via ZfitAnalysisState._poi_is_signal_strength

    # ------------------------------------------------------------------
    # Factory helpers (used by analysis_core.py)
    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, state: ZfitAnalysisState) -> "_ZfitAdapterWithState":
        """Return an adapter that carries signal_nominal_yield / poi_is_ss from state."""
        adapter = _ZfitAdapterWithState()
        adapter._state_ref = state
        return adapter


class _ZfitAdapterWithState(ZfitAnalysisBackend):
    """Thin subclass that reads metadata from a bound ZfitAnalysisState."""

    _state_ref: ZfitAnalysisState = None  # type: ignore[assignment]

    @property
    def signal_nominal_yield(self) -> Optional[float]:
        return getattr(self._state_ref, "_signal_nominal_yield", None)

    @property
    def poi_is_signal_strength(self) -> bool:
        return bool(getattr(self._state_ref, "_poi_is_signal_strength", True))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_hesse_error(result, poi_param) -> Optional[float]:
    try:
        hesse = result.hesse(params=[poi_param])
    except Exception:
        return None
    entry = hesse.get(poi_param)
    if not isinstance(entry, dict):
        return None
    error = entry.get("error")
    if error is None:
        return None
    return float(error)


def _extract_param_hesse_errors(fit_result) -> Dict[str, float]:
    params = [p for p in fit_result.params.keys() if getattr(p, "floating", False)]
    if not params:
        return {}
    try:
        hesse = fit_result.hesse(params=params)
    except Exception:
        return {}
    errors = {}
    for param in params:
        entry = hesse.get(param)
        if not isinstance(entry, dict):
            continue
        error = entry.get("error")
        if error is None:
            continue
        error = float(error)
        if np.isfinite(error) and error > 0.0:
            errors[param.name] = error
    return errors


def _extract_param_values(fit_result) -> Dict[str, float]:
    values = {}
    for param, info in fit_result.params.items():
        value = info.get("value") if isinstance(info, dict) else None
        if value is not None:
            values[param.name] = float(value)
    return values


def _build_loss(fit_model, resolved_fit_mode, binned_model, data):
    """Reconstruct the zfit loss for a given dataset.

    Delegates to the existing _build_loss in analysis_core to avoid
    duplicating the channel-model / binned-model branching logic during the
    transition period.
    """
    from zmodel.analysis_core import _build_loss as _core_build_loss
    return _core_build_loss(
        fit_model=fit_model,
        resolved_fit_mode=resolved_fit_mode,
        binned_model=binned_model,
        data=data,
    )
