"""Reference (numpy) implementations of Combine's yield and morphing functions.

These are the definitions every backend is validated against.  The formulas are ports of
Combine's C++ (interface/CombineMathFuncs.h, src/CMSHistFunc.cc, src/VerticalInterpHistPdf.cc)
at commit 137dbced of the mu2e_dev branch.
"""

import math

import numpy as np


def log_kappa_for_x(theta, log_kappa_lo, log_kappa_hi):
    """Combine's logKappaForX: smooth interpolation of log(kappa) for asymmetric lnN.

    For |theta| >= 0.5 it returns log(kappa_hi) (theta > 0) or -log(kappa_lo) (theta < 0);
    inside it uses h(2 theta) with h(x) = (3x^5 - 10x^3 + 15x)/8.
    """
    theta = np.asarray(theta, dtype=float)
    lkhi = log_kappa_hi
    lklo = -log_kappa_lo
    avg = 0.5 * (lkhi + lklo)
    halfdiff = 0.5 * (lkhi - lklo)
    twox = 2.0 * theta
    twox2 = twox * twox
    alpha = 0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0)
    inner = avg + alpha * halfdiff
    outer = np.where(theta >= 0, lkhi, lklo)
    return np.where(np.abs(theta) >= 0.5, outer, inner)


def asym_pow(theta, kappa_lo, kappa_hi):
    """Combine's asymPow: asymmetric log-normal response, equal to kappa_hi**theta for
    theta >= 0.5 and kappa_lo**(-theta) for theta <= -0.5."""
    return np.exp(log_kappa_for_x(theta, math.log(kappa_lo), math.log(kappa_hi)) * np.asarray(theta, dtype=float))


def smooth_step(x, vsmooth=1.0):
    """Combine's smoothStepFunc: 1 for x >= vsmooth, -1 for x <= -vsmooth, and
    (3x^5 - 10x^3 + 15x)/8 of x/vsmooth in between."""
    x = np.asarray(x, dtype=float)
    xn = x / vsmooth
    xn2 = xn * xn
    inner = 0.125 * xn * (xn2 * (3.0 * xn2 - 10.0) + 15.0)
    return np.where(np.abs(x) >= vsmooth, np.sign(x), inner)


def vertical_morph(nominal, ups, downs, thetas, vsmooth=1.0, floor=1e-9):
    """Vertical template morphing (Combine ``shape``) of *normalised* templates.

    ``nominal`` and each up/down are arrays with unit sum; the morphed template is::

        t = nominal + sum_i alpha_i,   alpha_i = x/2 * ((dhi - dlo) + (dhi + dlo) * S(x))

    with dhi = up - nominal, dlo = down - nominal and S the smooth step.  Bins are clipped at
    ``floor`` and the result renormalised to unit sum, as in FastVerticalInterpHistPdf2.
    """
    t = np.array(nominal, dtype=float)
    for up, down, x in zip(ups, downs, thetas):
        dhi = np.asarray(up, dtype=float) - nominal
        dlo = np.asarray(down, dtype=float) - nominal
        t = t + 0.5 * x * ((dhi - dlo) + (dhi + dlo) * smooth_step(x, vsmooth))
    t = np.where(t <= 0, floor, t)
    return t / t.sum()


def template_norm_kappas(nominal, up, down, scale=1.0):
    """(kappa_lo, kappa_hi) of the normalisation part of a template ``shape`` systematic, or
    None when Combine drops it (|kappa - 1| < 1e-3 for both).  Combine (ShapeTools.py) raises
    the integral ratios to the power of the datacard scale."""
    nom_int = float(np.sum(nominal))
    k_up = float(np.sum(up)) / nom_int
    k_down = float(np.sum(down)) / nom_int
    if abs(k_up - 1.0) < 1e-3 and abs(k_down - 1.0) < 1e-3:
        return None
    return k_down ** scale, k_up ** scale


def template_expected(contents, rate, systs, thetas):
    """Expected per-bin yields of a template process (no rate modifiers other than the
    template systematics).  ``systs`` is a list of (up, down, scale) raw templates and
    ``thetas`` the corresponding nuisance values.

    Combine semantics (ShapeTools.py, FastVerticalInterpHistPdf2):
      * the unit-normalised nominal is morphed with coefficient scale*theta, using a smooth
        region equal to min(1, min(scale));
      * the normalisation changes by asymPow(theta, kappa_lo**scale, kappa_hi**scale).
    """
    nom = np.asarray(contents, dtype=float)
    nom_int = nom.sum()
    if nom_int <= 0:
        return np.zeros_like(nom)
    ups, downs, xs, norm = [], [], [], 1.0
    vsmooth = 1.0
    for (up, down, scale), theta in zip(systs, thetas):
        up = np.asarray(up, dtype=float)
        down = np.asarray(down, dtype=float)
        ups.append(up / up.sum())
        downs.append(down / down.sum())
        xs.append(theta * scale)
        vsmooth = min(vsmooth, scale)
        kappas = template_norm_kappas(nom, up, down, scale)
        if kappas is not None:
            norm *= float(asym_pow(theta, kappas[0], kappas[1]))
    shape = vertical_morph(nom / nom_int, ups, downs, xs, vsmooth)
    return rate * norm * shape
