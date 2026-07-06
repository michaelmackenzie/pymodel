"""
pyhf backend adapter for the common analysis algorithms.

This module implements ``AnalysisBackend`` using pyhf for fitting and
hypothesis testing.

The *state* object used here is a ``PyhfAnalysisState`` dataclass containing
the pyhf model, the current data array (numpy), initial parameters, bounds,
fixed-parameter flags, and an RNG.  The common algorithms in
``backends/analysis_common.py`` treat this object as opaque and only pass it
back to the methods defined here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyhf

from backends.analysis_backend import AnalysisBackend
from backends.analysis_types import FitResult, HypothesisTestResult


# ---------------------------------------------------------------------------
# State object
# ---------------------------------------------------------------------------

@dataclass
class PyhfAnalysisState:
    """Mutable context bundling all pyhf fit-iteration state."""

    fit_model: Any
    """The hfmodel FitModel instance."""

    model: Any
    """The pyhf model object (fit_model.model)."""

    init_pars: List[float]
    """Initial parameter values (model.config.suggested_init() + overrides)."""

    par_bounds: List[Tuple[float, float]]
    """Parameter bounds (model.config.suggested_bounds() + overrides)."""

    fixed_params: List[bool]
    """Fixed-parameter mask (model.config.suggested_fixed() + overrides)."""

    rng: Any
    """numpy.random.Generator used for toy generation and jitter."""

    backend_name: str = "scipy"
    """pyhf numerical backend name (scipy / minuit / jax)."""

    hessian_method: str = "auto"
    """Method for Hessian uncertainty estimation."""

    # Current data – updated each iteration via set_data()
    current_data: Any = None

    # Snapshot of the observed data (never mutated)
    observed_data: Any = None

    # Metadata
    _signal_nominal_yield: Optional[float] = field(default=None, repr=False)
    _poi_is_signal_strength: bool = field(default=True, repr=False)


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------

class PyhfAnalysisBackend(AnalysisBackend):
    """AnalysisBackend implementation wrapping pyhf."""

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, state: PyhfAnalysisState) -> FitResult:
        if state.current_data is None:
            raise RuntimeError("PyhfAnalysisState.current_data is not set; call set_data() first")

        from hfmodel.analysis_core import _fit_with_retries, _estimate_hessian_uncertainties

        try:
            bestfit, unc, fit_status, fit_result_obj = _fit_with_retries(
                model=state.model,
                data=state.current_data,
                init_pars=state.init_pars,
                par_bounds=state.par_bounds,
                fixed_params=state.fixed_params,
                rng=state.rng,
                max_retries=4,
            )
        except Exception as exc:
            n = len(state.model.config.par_order)
            return FitResult(
                valid=False,
                poi_value=float("nan"),
                poi_uncertainty=None,
                nll=float("nan"),
                extra={"fit_error": str(exc)},
            )

        poi_index = int(state.model.config.poi_index)
        poi_value = float(bestfit[poi_index]) if np.isfinite(bestfit[poi_index]) else float("nan")

        # Hessian uncertainties
        hessian_unc, hessian_source = _estimate_hessian_uncertainties(
            model=state.model,
            data=state.current_data,
            bestfit=bestfit,
            par_bounds=state.par_bounds,
            fixed_params=state.fixed_params,
            result_obj=fit_result_obj,
            backend_name=state.backend_name,
            hessian_method=state.hessian_method,
        )

        poi_unc: Optional[float] = None
        if poi_index < len(unc) and np.isfinite(unc[poi_index]):
            poi_unc = float(unc[poi_index])
        elif hessian_unc is not None and poi_index < len(hessian_unc) and np.isfinite(hessian_unc[poi_index]):
            poi_unc = float(hessian_unc[poi_index])

        # NLL at best fit
        nll_val = float("nan")
        try:
            _, nll_raw = pyhf.infer.mle.fixed_poi_fit(
                poi_value,
                state.current_data,
                state.model,
                init_pars=state.init_pars,
                par_bounds=state.par_bounds,
                fixed_params=state.fixed_params,
                return_fitted_val=True,
            )
            nll_val = float(np.asarray(nll_raw, dtype=float).reshape(-1)[0])
        except Exception:
            fun = fit_status.get("fun") if isinstance(fit_status, dict) else None
            if fun is not None:
                nll_val = float(fun)

        par_names = list(state.model.config.par_order)
        param_values = {
            name: (float(bestfit[i]) if np.isfinite(bestfit[i]) else float("nan"))
            for i, name in enumerate(par_names)
        }
        param_uncertainties: Dict[str, float] = {}
        if hessian_unc is not None:
            for i, name in enumerate(par_names):
                if i < len(hessian_unc) and np.isfinite(hessian_unc[i]):
                    param_uncertainties[name] = float(hessian_unc[i])

        extra: Dict[str, Any] = {}
        if isinstance(fit_status, dict):
            extra["fit_status"] = fit_status
        if hessian_source is not None:
            extra["hessian_source"] = hessian_source

        return FitResult(
            valid=bool(np.all(np.isfinite(bestfit))),
            poi_value=poi_value,
            poi_uncertainty=poi_unc,
            nll=nll_val,
            param_values=param_values,
            param_uncertainties=param_uncertainties,
            extra=extra,
        )

    def fixed_poi_fit(self, state: PyhfAnalysisState, poi_value: float) -> FitResult:
        if state.current_data is None:
            raise RuntimeError("PyhfAnalysisState.current_data is not set")

        try:
            bestfit_raw, nll_raw = pyhf.infer.mle.fixed_poi_fit(
                float(poi_value),
                state.current_data,
                state.model,
                init_pars=state.init_pars,
                par_bounds=state.par_bounds,
                fixed_params=state.fixed_params,
                return_fitted_val=True,
            )
            bestfit = np.asarray(bestfit_raw, dtype=float)
            nll_val = float(np.asarray(nll_raw, dtype=float).reshape(-1)[0])
            valid = np.all(np.isfinite(bestfit)) and np.isfinite(nll_val)
        except Exception as exc:
            return FitResult(
                valid=False,
                poi_value=float(poi_value),
                poi_uncertainty=None,
                nll=float("nan"),
                extra={"fit_error": str(exc)},
            )

        return FitResult(
            valid=bool(valid),
            poi_value=float(poi_value),
            poi_uncertainty=None,
            nll=nll_val,
            param_values={
                name: (float(bestfit[i]) if i < len(bestfit) and np.isfinite(bestfit[i]) else float("nan"))
                for i, name in enumerate(state.model.config.par_order)
            },
        )

    def evaluate_nll(self, state: PyhfAnalysisState) -> float:
        if state.current_data is None:
            raise RuntimeError("PyhfAnalysisState.current_data is not set")
        poi_index = int(state.model.config.poi_index)
        # Use the snapshot POI value as the "current" value
        poi_val = float(state.init_pars[poi_index])
        try:
            val = pyhf.infer.mle.twice_nll(
                np.asarray(state.init_pars, dtype=float),
                np.asarray(state.current_data, dtype=float),
                state.model,
            )
            arr = np.asarray(val, dtype=float).reshape(-1)
            return float(arr[0]) if arr.size > 0 else float("nan")
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Hypothesis testing
    # ------------------------------------------------------------------

    def hypothesis_test(
        self,
        state: PyhfAnalysisState,
        poi_test: float,
        poi_alt: float = 0.0,
    ) -> HypothesisTestResult:
        if state.current_data is None:
            raise RuntimeError("PyhfAnalysisState.current_data is not set")

        try:
            result = pyhf.infer.hypotest(
                float(poi_test),
                state.current_data,
                state.model,
                test_stat="qtilde",
                return_expected_set=True,
                init_pars=state.init_pars,
                par_bounds=state.par_bounds,
                fixed_params=state.fixed_params,
            )
            obs = float(np.asarray(result[0], dtype=float).reshape(-1)[0])
            exp = np.asarray(result[1], dtype=float).reshape(-1)

            # pyhf returns [q2p5, q16, q50, q84, q97p5] (indices 0..4)
            expected_cls = {
                -2: float(exp[0]) if exp.size > 0 else float("nan"),
                -1: float(exp[1]) if exp.size > 1 else float("nan"),
                 0: float(exp[2]) if exp.size > 2 else float("nan"),
                 1: float(exp[3]) if exp.size > 3 else float("nan"),
                 2: float(exp[4]) if exp.size > 4 else float("nan"),
            }
        except Exception:
            obs = float("nan")
            expected_cls = {s: float("nan") for s in (-2, -1, 0, 1, 2)}

        return HypothesisTestResult(
            observed_cls=obs,
            expected_cls=expected_cls,
        )

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_poi_value(self, state: PyhfAnalysisState) -> float:
        poi_index = int(state.model.config.poi_index)
        return float(state.init_pars[poi_index])

    def set_poi_value(self, state: PyhfAnalysisState, value: float) -> None:
        poi_index = int(state.model.config.poi_index)
        state.init_pars[poi_index] = float(value)

    def poi_name(self, state: PyhfAnalysisState) -> str:
        return str(state.model.config.poi_name)

    def parameter_names(self, state: PyhfAnalysisState) -> List[str]:
        return list(state.model.config.par_order)

    def parameter_values(self, state: PyhfAnalysisState) -> Dict[str, float]:
        return {
            name: float(state.init_pars[i])
            for i, name in enumerate(state.model.config.par_order)
        }

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_parameters(self, state: PyhfAnalysisState) -> List[float]:
        return list(state.init_pars)

    def restore_parameters(self, state: PyhfAnalysisState, snapshot: List[float]) -> None:
        state.init_pars = list(snapshot)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_toy_data(self, state: PyhfAnalysisState) -> np.ndarray:
        from hfmodel.analysis_core import _generate_toy_data
        truth_pars = np.asarray(state.init_pars, dtype=float)
        return _generate_toy_data(state.model, truth_pars, state.rng)

    def generate_asimov_data(self, state: PyhfAnalysisState) -> np.ndarray:
        from hfmodel.analysis_core import _asimov_data
        truth_pars = np.asarray(state.init_pars, dtype=float)
        return _asimov_data(state.model, truth_pars)

    def get_observed_data(self, state: PyhfAnalysisState) -> np.ndarray:
        return np.asarray(state.fit_model.data, dtype=float)

    def get_current_data(self, state: PyhfAnalysisState) -> np.ndarray:
        if state.current_data is None:
            return self.get_observed_data(state)
        return np.asarray(state.current_data, dtype=float)

    def set_data(self, state: PyhfAnalysisState, data: Any) -> None:
        state.current_data = np.asarray(data, dtype=float)

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def poi_uncertainty_hesse(
        self, state: PyhfAnalysisState, fit_result: FitResult
    ) -> Optional[float]:
        # Already extracted during fit(); fall back to profile method if None.
        return fit_result.poi_uncertainty

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def signal_nominal_yield(self) -> Optional[float]:
        return None  # Overridden in _PyhfAdapterWithState

    @property
    def poi_is_signal_strength(self) -> bool:
        return True  # Overridden in _PyhfAdapterWithState

    @property
    def delta_nll_one_sigma(self) -> float:
        # pyhf stores twice_nll = -2 log L in FitResult.nll, so the 1-sigma
        # delta-NLL crossing is at 1.0 (not 0.5 as for true NLL backends).
        return 1.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, state: PyhfAnalysisState) -> "_PyhfAdapterWithState":
        adapter = _PyhfAdapterWithState()
        adapter._state_ref = state
        return adapter


class _PyhfAdapterWithState(PyhfAnalysisBackend):
    """Thin subclass that reads metadata from a bound PyhfAnalysisState."""

    _state_ref: PyhfAnalysisState = None  # type: ignore[assignment]

    @property
    def signal_nominal_yield(self) -> Optional[float]:
        return getattr(self._state_ref, "_signal_nominal_yield", None)

    @property
    def poi_is_signal_strength(self) -> bool:
        return bool(getattr(self._state_ref, "_poi_is_signal_strength", True))
