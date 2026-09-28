"""Toy and Asimov dataset generation with Combine's options.

Combine (``-t N`` with ``--expectSignal``, ``--toysFrequentist``, ``--bypassFrequentistFit``,
``--toysNoSystematics``) defines these modes, reproduced here:

* default ("hybrid"/prior):  for every toy the constrained nuisances are drawn from their
  constraint pdfs around the nominal global observables, the main data are generated at
  those values, and the global observables stay at their nominal values.
* ``frequentist``:  the nuisances are first fitted to the observed data with r fixed to
  ``expect_signal`` (skipped with ``bypass_fit``, which uses the pre-fit values).  Every toy
  is generated at those values and its global observables are drawn from the constraint
  pdfs centred on them; fits of the toy use the toy global observables.
* ``no_systematics``:  nuisances at their nominal values, nothing randomised.
* Asimov (``ntoys == -1``):  expected data at r = ``expect_signal`` and the nominal
  nuisances (default), or at the frequentist-fit values with matching global observables.

Hypothesis-test toys (HybridNew LHC modes) use ``generate_at``: nuisances at the given
(conditional-fit) values with frequentist global observables.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from inference.model import Dataset, MainData, global_obs_matching, observed_dataset
from modelspec import ir as I


@dataclass
class ToyConfig:
    ntoys: int = 0                 # -1 for Asimov
    expect_signal: float = 0.0
    frequentist: bool = False
    bypass_fit: bool = False
    no_systematics: bool = False

    def describe(self) -> str:
        if self.ntoys == -1:
            mode = "asimov" + (" (post-fit nuisances)" if self.frequentist and not self.bypass_fit else "")
        elif self.no_systematics:
            mode = "no-systematics"
        elif self.frequentist:
            mode = "frequentist" + (" (pre-fit, bypassing the fit)" if self.bypass_fit else "")
        else:
            mode = "prior-sampled nuisances (Combine default)"
        return f"{mode}, r = {self.expect_signal:g}"


def _draw_from_constraint(c: I.Constraint, center: float, lo: float, hi: float, rng, what: str):
    """Draw from a constraint term. what == "param": draw the parameter given the global
    observable ``center``; what == "gobs": draw the global observable given the parameter."""
    for _ in range(1000):
        if c.kind == I.CONSTRAINT_GAUSS:
            x = rng.normal(center, c.sigma_hi)
        elif c.kind == I.CONSTRAINT_BIFURGAUSS:
            # pick a side with probability proportional to its width, then a half-normal
            if rng.random() < c.sigma_lo / (c.sigma_lo + c.sigma_hi):
                x = center - abs(rng.normal(0.0, c.sigma_lo))
            else:
                x = center + abs(rng.normal(0.0, c.sigma_hi))
        elif c.kind == I.CONSTRAINT_POISSON:
            # param given N: the posterior of a Poisson mean with flat prior is Gamma(N+1);
            # global observable given n: Poisson(n)
            x = rng.gamma(center + 1.0) if what == "param" else float(rng.poisson(center))
        elif c.kind == I.CONSTRAINT_FLAT:
            x = rng.uniform(lo, hi)
        else:
            raise ValueError(c.kind)
        if what == "gobs" or lo <= x <= hi:
            return float(x)
    raise RuntimeError(f"could not sample inside [{lo}, {hi}]")


def generate_main(lik, values, rng) -> Dict[str, MainData]:
    """Poisson-fluctuated main data for every channel at ``values`` (binned: per flattened
    bin, also for N-D channels; unbinned: Poisson(nu_tot) events from the backend's
    ``sample_unbinned``, shape (n,) or (n, ndim))."""
    expected = lik.expected_counts(values)
    out = {}
    for ch in lik.model.channels:
        mu = np.clip(expected[ch.name], 0.0, None)
        if ch.data.kind == "unbinned":
            n = int(rng.poisson(mu.sum()))
            out[ch.name] = MainData(kind="unbinned", values=np.asarray(lik.sample_unbinned(values, ch.name, n, rng)))
        else:
            out[ch.name] = MainData(kind=ch.data.kind, counts=rng.poisson(mu).astype(float))
    return out


def asimov_main(lik, values) -> Dict[str, MainData]:
    """Expected data.  Unbinned channels get a weighted dataset at the bin centres of the
    observable binning (what Combine's generateAsimov does; N-D: at the centres of the
    product bins, as generateWithHisto does with a TH2/TH3)."""
    expected = lik.expected_counts(values)
    out = {}
    for ch in lik.model.channels:
        mu = np.asarray(expected[ch.name], dtype=float)
        if ch.data.kind == "unbinned":
            centers = ch.observable.bin_centers()
            out[ch.name] = MainData(kind="unbinned", values=centers[:, 0] if ch.observable.ndim == 1 else centers,
                                    weights=mu)
        else:
            out[ch.name] = MainData(kind=ch.data.kind, counts=mu)
    return out


def _prefit(lik, expect_signal):
    x = lik.nominal_values()
    x[lik.poi_index] = expect_signal
    return x


def frequentist_values(lik, fitter, observed: Dataset, expect_signal: float, bypass: bool):
    """Nuisance values used for frequentist generation: a fit to data with r fixed."""
    if bypass:
        return _prefit(lik, expect_signal), None
    res = fitter.fit(observed, fixed={lik.poi: expect_signal})
    return res.values, res


def generate_at(lik, values, rng, label="toy") -> Dataset:
    """A frequentist toy at ``values``: main data generated there and global observables drawn
    from the constraint pdfs centred on the parameter values."""
    gobs = {}
    for p in lik.model.parameters.values():
        c = p.constraint
        if c is None or not c.has_global_observable:
            continue
        theta = float(values[lik.index[p.name]])
        gobs[p.name] = theta if p.role == I.ROLE_CONSTANT else _draw_from_constraint(c, theta, p.lo, p.hi, rng, "gobs")
    return Dataset(main=generate_main(lik, values, rng), global_obs=gobs, label=label,
                   truth=lik.values_dict(values))


def asimov_at(lik, values, label="asimov") -> Dataset:
    return Dataset(main=asimov_main(lik, values), global_obs=global_obs_matching(lik.model, lik.index, values),
                   label=label, truth=lik.values_dict(values))


def toy_base(lik, fitter, cfg: ToyConfig, observed: Optional[Dataset] = None):
    """Parameter values the toys of ``cfg`` are generated around, and the run info."""
    observed = observed or observed_dataset(lik.model)
    info = {"mode": cfg.describe()}
    if cfg.frequentist:
        base, fit = frequentist_values(lik, fitter, observed, cfg.expect_signal, cfg.bypass_fit)
        if fit is not None:
            info["frequentist_fit"] = {"valid": fit.valid, "status": fit.status, "values": lik.values_dict(fit.values)}
            if not fit.valid:
                raise RuntimeError(f"frequentist fit to data failed ({fit.status}); use --bypass-frequentist-fit "
                                   "to generate from the pre-fit values")
    else:
        base = _prefit(lik, cfg.expect_signal)
    return base, info


def make_toy(lik, cfg: ToyConfig, base, rng, index: int) -> Dataset:
    """Toy number ``index`` of ``cfg`` (not Asimov), drawn with ``rng``."""
    if cfg.frequentist:
        return generate_at(lik, base, rng, label=f"toy_{index}")
    nominal_gobs = observed_dataset(lik.model).global_obs
    x = np.array(base, dtype=float)
    if not cfg.no_systematics:
        for p in lik.model.parameters.values():
            c = p.constraint
            if c is None or p.role == I.ROLE_CONSTANT:
                continue
            center = nominal_gobs.get(p.name, p.value)
            x[lik.index[p.name]] = _draw_from_constraint(c, center, p.lo, p.hi, rng, "param")
    return Dataset(main=generate_main(lik, x, rng), global_obs=dict(nominal_gobs), label=f"toy_{index}",
                   truth=lik.values_dict(x))


def generate_toys(lik, fitter, cfg: ToyConfig, rng, observed: Optional[Dataset] = None):
    """Generate datasets for ``fit``/``generate`` style toys.  Returns (datasets, info).

    ``rng`` is a ``ToySeeds``, an int seed or a numpy Generator (see ``inference.parallel``);
    toy i is drawn from its own stream (STREAM_GEN, i), so toy i does not depend on how many
    toys are generated or in which process."""
    from inference.parallel import PURPOSE_GENERATE, STREAM_GEN, as_seeds

    seeds = as_seeds(rng)
    observed = observed or observed_dataset(lik.model)
    base, info = toy_base(lik, fitter, cfg, observed)
    if cfg.ntoys == -1:
        if cfg.frequentist:
            return [asimov_at(lik, base)], info
        return [Dataset(main=asimov_main(lik, base), global_obs=observed_dataset(lik.model).global_obs,
                        label="asimov", truth=lik.values_dict(base))], info
    toys = [make_toy(lik, cfg, base, seeds.rng(STREAM_GEN, i, PURPOSE_GENERATE), i) for i in range(cfg.ntoys)]
    return toys, info


def toy_fit_task(ctx, task):
    """Worker task for ``fit -t`` / ``generate``: make toys ``start..stop-1`` of ``cfg`` around
    ``base`` (stream STREAM_GEN) and, with ``fit``, fit each one (fitter restarts seeded per
    toy).  Returns [(toy without native caches, FitResult or None)]."""
    from inference.parallel import PURPOSE_FREE_FIT, PURPOSE_GENERATE, STREAM_GEN, ToySeeds, portable, seeded_fitter

    lik, fitter = ctx.lik, ctx.fitter
    seeds = ToySeeds(task["seed"])
    cfg, base = task["cfg"], np.asarray(task["base"], dtype=float)
    out = []
    for i in range(task["start"], task["stop"]):
        toy = make_toy(lik, cfg, base, seeds.rng(STREAM_GEN, i, PURPOSE_GENERATE), i)
        res = None
        if task["fit"]:
            with seeded_fitter(fitter, seeds, STREAM_GEN, i, PURPOSE_FREE_FIT):
                res = fitter.fit(toy, fixed=task["fixed"], hesse=True, minos=task["minos"])
        out.append((portable(toy), res))
    return out
