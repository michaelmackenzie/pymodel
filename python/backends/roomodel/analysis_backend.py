"""
RooFit backend adapter for the common analysis algorithms (used by roomodel).

This module implements ``AnalysisBackend`` using RooFit for fitting and
hypothesis testing through the roomodel module.

The *state* object used here is a ``RooFitAnalysisState`` dataclass containing
the workspace, model, POI, data, and other RooFit-specific context. The common
algorithms in ``backends/analysis_common.py`` treat this object as opaque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backends.analysis_backend import AnalysisBackend
from backends.analysis_types import FitResult, HypothesisTestResult
from backends.analysis_common import is_signal_strength_poi


def _get_root():
    """Import ROOT and suppress RooFit messages."""
    import ROOT
    try:
        if not getattr(ROOT, "_pymodel_roofit_quiet", False):
            ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
            ROOT._pymodel_roofit_quiet = True
    except Exception:
        pass
    return ROOT


@dataclass
class RooFitAnalysisState:
    """Mutable context bundling all RooFit fit-iteration state."""

    workspace: Any
    """The RooWorkspace object."""

    model: Any
    """The model PDF (RooSimultaneous or RooProdPdf)."""

    poi: Any
    """The parameter of interest (RooRealVar)."""

    poi_name: str
    """Name of the POI."""

    fit_model: Any
    """The roomodel FitModel instance with metadata."""

    rng: Any
    """numpy.random.Generator used for toy generation."""

    # Current data – updated each iteration via set_data()
    current_data: Any = None

    # Snapshot of the observed data (never mutated)
    observed_data: Any = None

    # Snapshot for parameter restoration
    _param_snapshot: Optional[Any] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------

class RooFitAnalysisBackend(AnalysisBackend):
    """AnalysisBackend implementation wrapping RooFit (roomodel)."""

    def __init__(self, workspace: Any, model: Any, inner_model: Any, poi: Any, poi_name: str, fit_model: Any, observed_data: Any):
        """Initialize the RooFit backend adapter.

        Parameters
        ----------
        workspace : RooWorkspace
            The RooWorkspace containing all variables and PDFs.
        model : RooPdf
            The model PDF for fitting.
        inner_model : RooPdf
            Unwrapped model for toy generation (e.g., without constraint PDFs).
        poi : RooRealVar
            The parameter of interest.
        poi_name : str
            Name of the POI.
        fit_model : roomodel.FitModel
            The FitModel with process and signal information.
        observed_data : RooDataset or RooDataHist or None
            The observed data (may be None).
        """
        self.workspace = workspace
        self.model = model  # Constrained model used for fitting
        self.inner_model = inner_model  # Unwrapped model for toy generation
        self.poi = poi
        self._poi_name = poi_name
        self.fit_model = fit_model
        self.observed_data = observed_data
        self._signal_nominal_yield = None
        self._poi_is_signal_strength = is_signal_strength_poi(poi_name)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, state: RooFitAnalysisState) -> FitResult:
        """Run an unconstrained MLE fit and return the result."""
        ROOT = _get_root()
        if state.current_data is None:
            raise RuntimeError("RooFitAnalysisState.current_data is not set")

        try:
            # Import fitting utilities from roomodel
            from roomodel.analyze_model import _get_root as room_get_root
            
            # Fit with multiple strategies to find the best result
            fit_result = None
            best_result = None
            best_score = None

            can_extend = False
            try:
                can_extend = bool(self.model.canBeExtended())
            except Exception:
                can_extend = False

            recover_from_undef = getattr(ROOT.RooFit, "RecoverFromUndefinedRegions", None)

            for strategy in (0, 1, 2):
                fit_opts = [
                    ROOT.RooFit.Save(True),
                    ROOT.RooFit.PrintLevel(-1),
                    ROOT.RooFit.Strategy(int(strategy)),
                ]
                if can_extend:
                    fit_opts.append(ROOT.RooFit.Extended(True))
                if callable(recover_from_undef):
                    try:
                        fit_opts.append(recover_from_undef(10.0))
                    except Exception:
                        pass

                try:
                    trial = self.model.fitTo(state.current_data, *fit_opts)
                except Exception:
                    trial = None

                if trial is None or not bool(trial):
                    continue

                status_trial = int(trial.status())
                cov_trial = int(trial.covQual())
                score = (status_trial, -cov_trial)

                if best_result is None or score < best_score:
                    best_result = trial
                    best_score = score

                if status_trial == 0 and cov_trial >= 2:
                    fit_result = trial
                    break

            if fit_result is None:
                fit_result = best_result

            if fit_result is None:
                return FitResult(
                    valid=False,
                    poi_value=float("nan"),
                    poi_uncertainty=None,
                    nll=float("nan"),
                    extra={"fit_error": "No fit result obtained"},
                )

            status = int(fit_result.status())
            cov_qual = int(fit_result.covQual())
            valid = (status == 0) and (cov_qual >= 2)

            poi_value = float(self.poi.getVal())
            poi_unc = None
            try:
                poi_unc = float(self.poi.getError())
                if poi_unc <= 0.0:
                    poi_unc = None
            except Exception:
                poi_unc = None

            nll_val = float("nan")
            try:
                nll_val = float(fit_result.minNll())
            except Exception:
                nll_val = float("nan")

            return FitResult(
                valid=bool(valid),
                poi_value=poi_value,
                poi_uncertainty=poi_unc,
                nll=nll_val,
            )

        except Exception as exc:
            return FitResult(
                valid=False,
                poi_value=float("nan"),
                poi_uncertainty=None,
                nll=float("nan"),
                extra={"fit_error": str(exc)},
            )

    def fixed_poi_fit(self, state: RooFitAnalysisState, poi_value: float) -> FitResult:
        """Run an MLE fit with the POI fixed at *poi_value*."""
        ROOT = _get_root()
        if state.current_data is None:
            raise RuntimeError("RooFitAnalysisState.current_data is not set")

        try:
            old_const = bool(self.poi.isConstant())
            old_value = float(self.poi.getVal())

            self.poi.setVal(float(poi_value))
            self.poi.setConstant(True)

            try:
                can_extend = bool(self.model.canBeExtended())
            except Exception:
                can_extend = False

            fit_opts = [
                ROOT.RooFit.Save(True),
                ROOT.RooFit.PrintLevel(-1),
                ROOT.RooFit.Strategy(0),
            ]
            if can_extend:
                fit_opts.append(ROOT.RooFit.Extended(True))

            try:
                fit_result = self.model.fitTo(state.current_data, *fit_opts)
            except Exception:
                fit_result = None

            if fit_result is None or not bool(fit_result):
                return FitResult(
                    valid=False,
                    poi_value=float(poi_value),
                    poi_uncertainty=None,
                    nll=float("nan"),
                )

            nll_val = float("nan")
            try:
                nll_val = float(fit_result.minNll())
            except Exception:
                nll_val = float("nan")

            return FitResult(
                valid=bool(np.isfinite(nll_val)),
                poi_value=float(poi_value),
                poi_uncertainty=None,
                nll=nll_val,
            )

        except Exception as exc:
            return FitResult(
                valid=False,
                poi_value=float(poi_value),
                poi_uncertainty=None,
                nll=float("nan"),
                extra={"fit_error": str(exc)},
            )
        finally:
            # Restore POI state
            try:
                self.poi.setVal(old_value)
                self.poi.setConstant(old_const)
            except Exception:
                pass

    def evaluate_nll(self, state: RooFitAnalysisState) -> float:
        """Return the NLL evaluated at the current parameter values."""
        ROOT = _get_root()
        try:
            if state.current_data is None:
                return float("nan")

            can_extend = False
            try:
                can_extend = bool(self.model.canBeExtended())
            except Exception:
                can_extend = False

            nll_opts = [ROOT.RooFit.Offset(True)]
            if can_extend:
                nll_opts.append(ROOT.RooFit.Extended(True))

            nll = self.model.createNLL(state.current_data, *nll_opts)
            if nll is None or not bool(nll):
                return float("nan")

            nll_val = float(nll.getVal())
            return nll_val if np.isfinite(nll_val) else float("nan")

        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Hypothesis testing (not needed for FC, but required by protocol)
    # ------------------------------------------------------------------

    def hypothesis_test(self, state: RooFitAnalysisState, poi_test: float, poi_alt: float = 0.0) -> HypothesisTestResult:
        """Not implemented for roomodel FC mode."""
        raise NotImplementedError("hypothesis_test is not used in roomodel FC workflow")

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_poi_value(self, state: RooFitAnalysisState) -> float:
        """Return the current value of the POI."""
        try:
            return float(self.poi.getVal())
        except Exception:
            return float("nan")

    def set_poi_value(self, state: RooFitAnalysisState, value: float) -> None:
        """Set the POI to a specific value."""
        try:
            self.poi.setVal(float(value))
        except Exception:
            pass

    def poi_name(self, state: RooFitAnalysisState) -> str:
        """Return the name of the POI."""
        return self._poi_name

    def parameter_names(self, state: RooFitAnalysisState) -> List[str]:
        """Return names of all floating parameters."""
        names = []
        try:
            params = self.model.getParameters(state.current_data.get() if state.current_data else None)
            for var in self._iter_roo_collection(params):
                if not var.InheritsFrom("RooRealVar"):
                    continue
                try:
                    if not bool(var.isConstant()):
                        names.append(str(var.GetName()))
                except Exception:
                    pass
        except Exception:
            pass
        return names

    def parameter_values(self, state: RooFitAnalysisState) -> Dict[str, float]:
        """Return {name: value} for all parameters."""
        values = {}
        try:
            params = self.model.getParameters(state.current_data.get() if state.current_data else None)
            for var in self._iter_roo_collection(params):
                if not var.InheritsFrom("RooRealVar"):
                    continue
                try:
                    name = str(var.GetName())
                    values[name] = float(var.getVal())
                except Exception:
                    pass
        except Exception:
            pass
        return values

    # ------------------------------------------------------------------
    # Parameter snapshots
    # ------------------------------------------------------------------

    def snapshot_parameters(self, state: RooFitAnalysisState) -> Any:
        """Capture a snapshot of current parameter state."""
        snapshot = {}
        try:
            all_vars = self.workspace.allVars()
            for var in self._iter_roo_collection(all_vars):
                if not var.InheritsFrom("RooRealVar"):
                    continue
                try:
                    name = str(var.GetName())
                    snapshot[name] = {
                        "value": float(var.getVal()),
                        "constant": bool(var.isConstant()),
                    }
                except Exception:
                    pass
        except Exception:
            pass
        return snapshot

    def restore_parameters(self, state: RooFitAnalysisState, snapshot: Any) -> None:
        """Restore parameter state from a snapshot."""
        if not snapshot:
            return
        try:
            for name, cfg in snapshot.items():
                var = self.workspace.var(name)
                if var is None or not bool(var):
                    continue
                try:
                    var.setVal(float(cfg.get("value")))
                except Exception:
                    pass
                try:
                    var.setConstant(bool(cfg.get("constant")))
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_toy_data(self, state: RooFitAnalysisState) -> Any:
        """Generate a toy dataset at current parameter values."""
        from roomodel.analyze_model import _generate_dataset, _resolve_obs_var

        try:
            obs_var = _resolve_obs_var(state.current_data, self.workspace)
            toy_data = _generate_dataset(
                self.workspace,
                self.model,
                state.current_data,
                obs_var,
                fit_mode="unbinned",
                binned_bins=40,
            )
            return toy_data
        except Exception:
            return None

    def generate_asimov_data(self, state: RooFitAnalysisState) -> Any:
        """Return an Asimov (expected) dataset."""
        # Not needed for FC in roomodel
        raise NotImplementedError("generate_asimov_data not implemented for roomodel")

    def get_observed_data(self, state: RooFitAnalysisState) -> Any:
        """Return the observed dataset."""
        return self.observed_data

    def get_current_data(self, state: RooFitAnalysisState) -> Any:
        """Return the currently loaded dataset."""
        return state.current_data

    def set_data(self, state: RooFitAnalysisState, data: Any) -> None:
        """Set the current dataset."""
        state.current_data = data

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def poi_uncertainty_hesse(self, state: RooFitAnalysisState, fit_result: FitResult) -> Optional[float]:
        """Return POI uncertainty from Hessian."""
        return fit_result.poi_uncertainty

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def signal_nominal_yield(self) -> Optional[float]:
        """Nominal signal yield for converting mu limits to yield limits."""
        return self._signal_nominal_yield

    @property
    def poi_is_signal_strength(self) -> bool:
        """True if POI is a signal-strength modifier."""
        return self._poi_is_signal_strength

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_roo_collection(collection):
        """Iterate over a RooFit collection."""
        if collection is None:
            return
        try:
            for obj in collection:
                yield obj
        except TypeError:
            return
