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


# ----------------------------------------------------------------------------------------
# autoMCStats (Barlow-Beeston-lite): CMSHistFunc templates and CMSHistErrorPropagator
# ----------------------------------------------------------------------------------------

CMSHIST_FLOOR = 1e-9  # FastTemplate::CropUnderflows default, applied by CMSHistFunc and the propagator
BB_SIGMA_RANGE = 7.0  # Gaussian bin parameters live in [-7, 7]; Poisson ranges cover +-7 sigma


def cmshist_template(contents, systs, thetas):
    """(h, norm) of a TH1 template in an autoMCStats channel (Combine CMSHistFunc,
    ShapeTools.getPdf with hist-mode 1, src/CMSHistFunc.cc updateCache).

    ``h`` is the *un-normalised* nominal (its integral is the process rate), vertically
    morphed with the up/down templates rescaled to the nominal integral::

        h = h0 + sum_k x_k/2 ((u_k - d_k) + (u_k + d_k - 2 h0) S(x_k)),   x_k = scale_k theta_k

    (S = smooth_step with region min(1, scales)), then floored at 1e-9 per bin and NOT
    renormalised (unlike FastVerticalInterpHistPdf2).  ``norm`` is the product of the usual
    asymPow(theta, kd**scale, ku**scale) template normalisation terms (getExtraNorm)."""
    nom = np.asarray(contents, dtype=float)
    h = nom.copy()
    norm = 1.0
    if systs:
        nom_int = float(nom.sum())
        vsmooth = min([1.0] + [scale for _, _, scale in systs])
        for (up, down, scale), theta in zip(systs, thetas):
            up = np.asarray(up, dtype=float)
            down = np.asarray(down, dtype=float)
            upr = up * (nom_int / up.sum()) if up.sum() > 0 else up
            downr = down * (nom_int / down.sum()) if down.sum() > 0 else down
            x = theta * scale
            dhi, dlo = upr - nom, downr - nom
            h = h + 0.5 * x * ((dhi - dlo) + (dhi + dlo) * smooth_step(x, vsmooth))
            kappas = template_norm_kappas(nom, up, down, scale)
            if kappas is not None:
                norm *= float(asym_pow(theta, kappas[0], kappas[1]))
    return np.maximum(h, CMSHIST_FLOOR), norm


def poisson_bb_range(n_eff: float):
    """[lo, hi] of a Poisson bin parameter with nominal n_eff (setupBinPars: 7-sigma
    chi-square quantiles)."""
    from scipy.stats import chi2, norm

    p = norm.sf(BB_SIGMA_RANGE)
    return 0.5 * float(chi2.ppf(p, 2.0 * n_eff)), 0.5 * float(chi2.ppf(1.0 - p, 2.0 * n_eff + 2.0))


def bb_lite_classify(channel: str, procs, threshold: float, include_signal: bool):
    """Combine's CMSHistErrorPropagator::setupBinPars at the nominal parameters.

    ``procs``: list of (name, is_signal, C0, h0, e) with C0 the nominal normalisation
    coefficient, h0 the nominal CMSHistFunc template and e = sqrt(sumw2) per bin.  Returns a
    list of (bin, kind, process, n_eff, param_name) in Combine's parameter order:

    * signal processes are left out of the n_eff decision unless ``include_signal`` (they
      still enter the error sums and get per-process parameters);
    * sub_sum = sum C0 h0, sub_err = sqrt(sum (C0 e)^2) over those processes; a bin with
      sub_err <= 0 gets no parameter;
    * n = floor(0.5 + sub_sum^2 / sub_err^2); if n > threshold (and the total error of all
      processes is > 0): one Gaussian parameter ``prop_bin<ch>_bin<i>``;
    * otherwise per process (``prop_bin<ch>_bin<i>_<proc>``), with v = h0, e as above:
      e <= 0 or v < 0: none; v > 0 and v >= 0.999 e: n_p = floor(0.5 + v^2/e^2), Poisson if
      n_p <= threshold, else Gaussian; v >= 0 and e > v: Gaussian.
    """
    nbins = len(procs[0][3]) if procs else 0
    out = []
    for j in range(nbins):
        sub_sum = sub_err2 = tot_err2 = 0.0
        for name, is_sig, c0, h0, e in procs:
            tot_err2 += (e[j] * c0) ** 2
            if is_sig and not include_signal:
                continue
            sub_sum += h0[j] * c0
            sub_err2 += (e[j] * c0) ** 2
        if not sub_err2 > 0.0:
            continue
        n = math.floor(0.5 + sub_sum * sub_sum / sub_err2)
        if n <= threshold:
            for name, is_sig, c0, h0, e in procs:
                v, ep = float(h0[j]), float(e[j])
                pname = f"prop_bin{channel}_bin{j}_{name}"
                if ep <= 0.0 or v < 0.0:
                    continue
                if v > 0.0 and v >= ep * 0.999:
                    n_p = math.floor(0.5 + v * v / (ep * ep))
                    if n_p <= threshold:
                        out.append((j, "poisson", name, float(n_p), pname))
                    else:
                        out.append((j, "gauss", name, 0.0, pname))
                else:
                    out.append((j, "gauss", name, 0.0, pname))
        elif tot_err2 > 0.0:
            out.append((j, "total", "", 0.0, f"prop_bin{channel}_bin{j}"))
    return out


def bb_lite_expected(coeffs, hists, errors, params, values):
    """Expected yields of an autoMCStats channel (CMSHistErrorPropagator::updateCache).

    ``coeffs[p]`` = C_p (normalisation coefficient at the current parameters: r, norm terms,
    template asymPow), ``hists[p]`` = h_p (``cmshist_template``), ``errors[p]`` = e_p,
    ``params``: the channel's MCStatsParam list, ``values[name]``: parameter values.
    Per bin i (processes p):

        total:   nu_i = sum_p C_p h_pi + x_i * sqrt(sum_p (C_p e_pi)^2)
        poisson: nu_pi = C_p h_pi * gamma / n_eff      (= C_p h_pi + (gamma/n_eff - 1) C_p h_pi)
        gauss:   nu_pi = C_p h_pi + x * C_p e_pi

    and the channel total is floored at 1e-9 per bin (cache_.CropUnderflows()).  Returns
    (per-process yields, total).  For "total" bins the shift x*sigma_i is attributed to the
    processes in proportion to (C_p e_pi)^2 (as Combine's CMSHistFuncWrapper does), so the
    per-process yields sum to the unfloored total."""
    by = {p: float(coeffs[p]) * np.asarray(hists[p], dtype=float) for p in hists}
    for bp in params:
        j = bp.bin
        val = float(values[bp.param])
        if bp.kind == "total":
            w = {p: (float(coeffs[p]) * float(errors[p][j])) ** 2 for p in hists}
            s2 = sum(w.values())
            if s2 > 0.0:
                sigma = math.sqrt(s2)
                for p in hists:
                    by[p][j] += val * sigma * w[p] / s2
        elif bp.kind == "poisson":
            p = bp.process
            by[p][j] += (val / bp.n_eff - 1.0) * float(coeffs[p]) * float(hists[p][j])
        elif bp.kind == "gauss":
            p = bp.process
            by[p][j] += val * float(errors[p][j]) * float(coeffs[p])
        else:
            raise ValueError(f"unknown autoMCStats parameter kind {bp.kind}")
    total = np.maximum(np.sum(list(by.values()), axis=0), CMSHIST_FLOOR)
    return by, total


# ----------------------------------------------------------------------------------------
# shape systematics on RooAbsPdfs (ShapeTools.getPdf)
# ----------------------------------------------------------------------------------------

PDF_FLOOR = 1e-15           # VerticalInterpPdf::_pdfFloorVal (un-normalised value <= 0)
PDF_INTEGRAL_FLOOR = 1e-10  # VerticalInterpPdf::_integralFloorVal (integral <= 0)


def vertical_pdf_coefficients(x, qrange):
    """(c_cen, c_up, c_dn) of VerticalInterpPdf::interpolate (quadratic algorithm 0, the
    ``shape`` default) for the coefficient x = scale * theta, such that the morph term is
    c_up * f_up + c_dn * f_dn + c_cen * f_0:

      |x| >= qrange:  x * (f_up - f_0) for x > 0,  x * (f_0 - f_dn) for x < 0  (linear)
      |x| <  qrange:  c_up = x (q + x) / 2q,  c_dn = -x (q - x) / 2q,  c_cen = -x^2 / q
    """
    x = float(x)
    if abs(x) >= qrange:
        return (-x, x, 0.0) if x > 0 else (x, 0.0, -x)
    return (-x * x / qrange, x * (qrange + x) / (2.0 * qrange), -x * (qrange - x) / (2.0 * qrange))


def vertical_pdf_fractions(contents, raw_integral, systs, thetas, widths, method="center"):
    """Bin fractions of Combine's VerticalInterpPdf (``shape`` on a RooAbsPdf that is not a
    RooHistPdf) for fixed pdfs.

    VerticalInterpPdf morphs the *un-normalised* pdf values f_j(x) = I_j p_j(x) (I_j the
    integral of the un-normalised pdf over the observable, p_j the normalised pdf) and
    normalises by the same morph of the integrals::

        density(x) = F(x) / N,   F(x) = f_0(x) + sum_k [c_up f_up,k + c_dn f_dn,k + c_cen f_0](x)
                                 N    = I_0    + sum_k [c_up I_up,k + c_dn I_dn,k + c_cen I_0]

    with the coefficients of ``vertical_pdf_coefficients`` (x_k = scale_k * theta_k, region
    q = min(1, scales)), F <= 0 -> 1e-15 and N <= 0 -> 1e-10.  ``contents`` (and the up/down
    contents in ``systs``: (up_contents, down_contents, up_integral, down_integral, scale))
    are the normalised pdfs on the bins, as in Shape.contents:

    * method "center":   fraction_i = density(x_i) * w_i  (contents = p(x_i) w_i)
    * method "integral": fraction_i = F_i / N, F_i the morph of the bin integrals, and
      F_i / N <= 0 -> 1e-10 (VerticalInterpPdf::analyticalIntegralWN over a range)
    """
    widths = np.asarray(widths, dtype=float)
    q = min([1.0] + [s[4] for s in systs])
    a0 = 1.0
    num = np.zeros_like(widths)
    den = 0.0
    for (up, down, i_up, i_down, scale), theta in zip(systs, thetas):
        c_cen, c_up, c_dn = vertical_pdf_coefficients(scale * theta, q)
        a0 += c_cen
        num = num + c_up * i_up * np.asarray(up, dtype=float) + c_dn * i_down * np.asarray(down, dtype=float)
        den += c_up * i_up + c_dn * i_down
    num = num + a0 * raw_integral * np.asarray(contents, dtype=float)
    den += a0 * raw_integral
    if method == "center":
        value = num / widths
        value = np.where(value > 0, value, PDF_FLOOR)
        return value / (den if den > 0 else PDF_INTEGRAL_FLOOR) * widths
    if method == "integral":
        ratio = num / den if den > 0 else np.zeros_like(num)
        return np.where(ratio > 0, ratio, PDF_INTEGRAL_FLOOR)
    raise ValueError(f"unknown bin integration '{method}'")


def hist_pdf_fractions(contents, systs, thetas, widths, floor=1e-9):
    """Bin fractions of Combine's FastVerticalInterpHistPdf2 built from fixed RooHistPdfs
    (``shape`` on a RooHistPdf nominal): vertical template morphing of the unit-normalised
    histograms (``contents`` and the (up, down, scale) of ``systs``, same binning) with
    Combine's smooth step (region min(1, scales)), cropped at a *density* of ``floor``
    (FastTemplate::CropUnderflows: values below 1e-9 per unit of x are set to 1e-9) and
    renormalised.  There is no normalisation effect (unlike TH1 templates).

    Combine samples every RooHistPdf into a TH1F (RooAbsReal::createHistogram), so the
    histograms are rounded to single precision first, as there."""
    widths = np.asarray(widths, dtype=float)
    nom = np.asarray(contents, dtype=np.float32).astype(float)
    nom = nom / nom.sum()
    t = nom.copy()
    q = min([1.0] + [s[2] for s in systs])
    for (up, down, scale), theta in zip(systs, thetas):
        up = np.asarray(up, dtype=np.float32).astype(float)
        down = np.asarray(down, dtype=np.float32).astype(float)
        dhi = up / up.sum() - nom
        dlo = down / down.sum() - nom
        x = scale * theta
        t = t + 0.5 * x * ((dhi - dlo) + (dhi + dlo) * smooth_step(x, q))
    t = np.where(t / widths < floor, floor * widths, t)
    return t / t.sum()
