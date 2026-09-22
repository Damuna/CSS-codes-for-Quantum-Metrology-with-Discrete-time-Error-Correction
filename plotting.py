"""Plotting and TeX-table routines for the manuscript."""

import os
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, lsq_linear
from scipy.signal import find_peaks

from analysis import (
    discrete_x_projector_decay_rate_series,
    discrete_x_spectral_decay_rates,
)

GHZ_COLOR = "black"
MASTER_COLOR = "black"
CROSSOVER_COLOR = "0.35"

TAU_COLORS = [
    "#3B4CC0",
    "#D45087",
    "#2A9D8F",
    "#F4A261",
    "#7A5195",
    "#6C757D",
]

TAU_COLOR_MAP = {
    1e-3: TAU_COLORS[0],
    5e-2: TAU_COLORS[1],
    1e-1: TAU_COLORS[2],
}

Z_EPS_COLOR_MAP = {
    1e-3: TAU_COLORS[0],
    1e-2: TAU_COLORS[1],
    1e-1: TAU_COLORS[2],
}

NO_QEC_COLOR = TAU_COLORS[3]
REFERENCE_GRAY = "#6C757D"

UNIVERSAL_TAU_COLORS_ST = {
    1e-3: "#3B4CC0",
    5e-2: "#D45087",
    1e-1: "#2A9D8F",
}

UNIVERSAL_TAU_COLORS_SH = {
    1e-3: "#35B8E8",
    5e-2: "#FF6FAE",
    1e-1: "#41D6B3",
}


def paper_style():
    matplotlib.rcParams.update(
        {
            "ytick.color": "black",
            "xtick.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Serif"],
            "font.size": 20,
            "axes.titlesize": 18,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )


def _decorate(ax):
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.set_facecolor("#F9F9FB")


def _save(fig, plot_dir, name):
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, name)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return path


def _unique_handles_labels(axes):
    unique = OrderedDict()
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in unique and label != "_nolegend_":
                unique[label] = handle
    return list(unique.values()), list(unique.keys())


def shared_legend(fig, axes, ncol=4, y=1.04):
    handles, labels = _unique_handles_labels(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=min(ncol, len(handles)),
            fancybox=False,
            shadow=False,
            framealpha=0.95,
        )


def _tau_label(tau):
    return rf"$\tau={float(tau):g}\,\mathrm{{s}}$"


def _epsilon_label(eps):
    return rf"$\epsilon={float(eps):g}\,\mathrm{{s}}^{{-1}}$"


def _tau_color(tau):
    tau = float(tau)
    for value, color in TAU_COLOR_MAP.items():
        if np.isclose(tau, value, rtol=1e-12, atol=1e-15):
            return color
    return colors_["PrussianBlue"]


def _z_eps_color(eps):
    eps = float(eps)
    for value, color in Z_EPS_COLOR_MAP.items():
        if np.isclose(eps, value, rtol=1e-12, atol=1e-15):
            return color
    return colors_["PrussianBlue"]


def _peak_data(t, y, skip=1, stride=2):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    peaks = find_peaks(y)[0]
    if stride > 1:
        peaks = peaks[stride * skip :: stride]
    else:
        peaks = peaks[skip:]
    tp, yp = t[peaks], y[peaks]
    valid = np.isfinite(tp) & np.isfinite(yp) & (tp > 0.0) & (yp > 0.0)
    return tp[valid], yp[valid]


def _stroboscopic_samples(t, p, tau, max_points=3000):
    nmax = int(np.floor(float(t[-1]) / float(tau)))
    n = np.arange(nmax + 1)
    if len(n) > max_points:
        idx = np.unique(np.rint(np.linspace(0, nmax, max_points)).astype(int))
        n = n[idx]
    ts = n * float(tau)
    ps = np.interp(ts, t, p)
    return ts, ps


def _dominant_frequency(groups):
    modes = [m for g in groups for m in g["modes"] if abs(m["frequency"]) > 1e-8]
    if not modes:
        return 0.0
    return float(max(modes, key=lambda m: abs(m["overlap"]))["frequency"])


def _spectral_projector_rates(code, spectral):
    groups = spectral["groups"]

    if code.lower() == "steane":

        def max_frequency(group):
            return max(abs(m["frequency"]) for m in group["modes"])

        real_group = min(groups, key=max_frequency)
        osc_group = max(groups, key=max_frequency)
        return {
            "real": float(real_group["decay_rate"]),
            "osc": float(osc_group["decay_rate"]),
        }

    if code.lower() == "shor":
        group = max(groups, key=lambda g: g["total_overlap"])
        return {"common": float(group["decay_rate"])}

    raise ValueError(code)


def fit_projector_qec(code, t, p, tau, omega, eps):
    """Fit the exact stroboscopic spectral form."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    ts, ps = _stroboscopic_samples(t, p, tau)

    spectral = discrete_x_spectral_decay_rates(code, omega, eps, tau)
    exact = _spectral_projector_rates(code, spectral)
    tau2 = discrete_x_projector_decay_rate_series(code, omega, eps, tau, order=2)
    Om0 = abs(_dominant_frequency(spectral["groups"]))
    Om_lo = 0.5 * Om0 if Om0 > 0.0 else 0.0
    Om_hi = 1.5 * Om0 if Om0 > 0.0 else np.inf

    if code.lower() == "steane":

        def model(x, Ar, Ao, Dr, Do, Om, phi):
            return (
                1.0 / 8.0
                + Ar * np.exp(-Dr * x)
                + Ao * np.exp(-Do * x) * np.cos(Om * x + phi)
            )

        p0 = (
            21.0 / 32.0,
            7.0 / 32.0,
            max(tau2["real"], 1e-14),
            max(tau2["osc"], 1e-14),
            Om0,
            0.0,
        )
        lo = (0.0, 0.0, 0.0, 0.0, Om_lo, -np.pi)
        hi = (1.0, 1.0, np.inf, np.inf, Om_hi, np.pi)
        pars, cov = curve_fit(model, ts, ps, p0=p0, bounds=(lo, hi), maxfev=200000)
        prediction = model(ts, *pars)
        residual = prediction - ps
        info = {
            "params": np.asarray(pars, float),
            "stderr": np.sqrt(np.maximum(np.diag(cov), 0.0)),
            "fit": {"real": float(pars[2]), "osc": float(pars[3])},
            "exact": exact,
            "tau2": tau2,
            "quality": {
                "n_points": int(len(ts)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "max_abs_error": float(np.max(np.abs(residual))),
            },
        }
        return ts, ps, lambda x: model(np.asarray(x), *pars), info

    if code.lower() == "shor":

        def model(x, A0, Ao, D, Om, phi):
            return 1.0 / 4.0 + np.exp(-D * x) * (A0 + Ao * np.cos(Om * x + phi))

        p0 = (
            3.0 / 8.0,
            3.0 / 8.0,
            max(tau2["common"], 1e-14),
            Om0,
            0.0,
        )
        lo = (0.0, 0.0, 0.0, Om_lo, -np.pi)
        hi = (1.0, 1.0, np.inf, Om_hi, np.pi)
        pars, cov = curve_fit(model, ts, ps, p0=p0, bounds=(lo, hi), maxfev=200000)
        prediction = model(ts, *pars)
        residual = prediction - ps
        info = {
            "params": np.asarray(pars, float),
            "stderr": np.sqrt(np.maximum(np.diag(cov), 0.0)),
            "fit": {"common": float(pars[2])},
            "exact": exact,
            "tau2": tau2,
            "quality": {
                "n_points": int(len(ts)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "max_abs_error": float(np.max(np.abs(residual))),
            },
        }
        return ts, ps, lambda x: model(np.asarray(x), *pars), info

    raise ValueError(code)


def _sensitivity_prefactor(code):
    if code.lower() == "steane":
        return 2.0 * np.sqrt(7.0)
    if code.lower() == "shor":
        return 6.0 * np.sqrt(3.0)
    raise ValueError(code)


def _sensitivity_factors_from_projector_rates(code, rates, tau):
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    if code.lower() == "steane":
        Dr = float(rates["real"])
        Do = float(rates["osc"])
        radicand = (3.0 * Dr + Do) / tau
        return {
            "half": float(3.0 / (4.0 * np.sqrt(2.0)) * np.sqrt(max(radicand, 0.0))),
            "one": float(Do / (2.0 * tau)),
        }

    if code.lower() == "shor":
        D = float(rates["common"])
        return {
            "half": float(0.5 * np.sqrt(max(D / tau, 0.0))),
            "one": float(D / (2.0 * tau)),
        }

    raise ValueError(code)


def fit_two_factor_envelope(code, t, y, tau):
    r"""Fit C t exp[-a_half sqrt(tau t)-a_one tau t].

    C is fixed to its noiseless value. The fitted line in the X-noise
    envelope plot is obtained by evaluating this returned object directly.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    valid = np.isfinite(t) & np.isfinite(y) & (t > 0.0) & (y > 0.0)
    t, y = t[valid], y[valid]
    if len(t) < 3:
        raise ValueError("Too few envelope points for the two-factor fit.")

    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    C = _sensitivity_prefactor(code)
    x = np.sqrt(tau * t)
    design = np.column_stack((x, x**2))
    target = -np.log(y / (C * t))

    # Both coefficients are decay factors. The bounds exclude the unphysical
    # negative rates produced when an unconstrained polynomial fit uses one
    # column to cancel another.
    scales = np.maximum(np.linalg.norm(design, axis=0), np.finfo(float).eps)
    result = lsq_linear(
        design / scales,
        target,
        bounds=(0.0, np.inf),
        lsmr_tol="auto",
        max_iter=10000,
    )
    if not result.success:
        raise RuntimeError(f"Two-factor fit failed: {result.message}")
    pars = result.x / scales

    log_prediction = -(design @ pars)
    log_target = np.log(y / (C * t))
    residual = log_prediction - log_target
    prediction = C * t * np.exp(log_prediction)
    relative = prediction / y - 1.0

    dof = max(len(t) - 2, 1)
    sigma2 = float(np.sum(residual**2) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return {
        "code": code,
        "tau": tau,
        "prefactor": float(C),
        "a_half": float(pars[0]),
        "a_one": float(pars[1]),
        "stderr": np.asarray(stderr, float),
        "covariance": np.asarray(covariance, float),
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "rms_relative_error": float(np.sqrt(np.mean(relative**2))),
        "max_relative_error": float(np.max(np.abs(relative))),
        "active_bounds": np.asarray(result.active_mask, int),
        "n_points": int(len(t)),
    }


def eval_two_factor_envelope(t, fit):
    t = np.asarray(t, float)
    tau = float(fit["tau"])
    exponent = -float(fit["a_half"]) * np.sqrt(tau * t) - float(fit["a_one"]) * tau * t
    return float(fit["prefactor"]) * t * np.exp(exponent)


def _two_factor_optimum_from_factors(a_half, a_one, tau):
    r"""Continuous-budget approximation obtained after removing the floor.

    For a fixed design budget T, replacing floor(T/t) by T/t makes the
    maximizer independent of T and gives the closed-form estimate used in the
    optimal-time table.  The coefficients supplied here are taken from the
    exact one-cycle matrix spectrum.
    """
    a_half = float(a_half)
    a_one = float(a_one)
    tau = float(tau)

    if (
        not np.isfinite(a_half)
        or not np.isfinite(a_one)
        or not np.isfinite(tau)
        or tau <= 0.0
        or a_half < 0.0
        or a_one < 0.0
    ):
        return np.nan

    denominator = a_half + np.sqrt(a_half * a_half + 8.0 * a_one)
    if denominator <= 0.0:
        return np.inf

    return float(4.0 / (tau * denominator * denominator))


def _numerical_budget_optimum(tp, yp, T):
    r"""Optimize sqrt(floor(T/t)) times the numerical envelope maxima.

    T is the fixed design budget, taken in this analysis to be the end of the
    simulated interval.  The maximization is performed over all available
    numerical envelope maxima with t_j <= T.

    If the best sampled maximum is the final available maximum, the turnover
    is not resolved inside the simulated interval.  In that boundary-limited
    case we report the physical boundary T.
    """
    tp = np.asarray(tp, float)
    yp = np.asarray(yp, float)
    T = float(T)
    valid = np.isfinite(tp) & np.isfinite(yp) & (tp > 0.0) & (yp > 0.0) & (tp <= T)
    tp_use, yp_use = tp[valid], yp[valid]
    if not len(tp_use):
        return {
            "time": T,
            "resolved": False,
            "peak_time": np.nan,
            "peak_value": np.nan,
            "objective": np.nan,
        }

    repetitions = np.maximum(np.floor(T / tp_use + 1e-12).astype(int), 1)
    objective = np.sqrt(repetitions) * yp_use
    i = int(np.nanargmax(objective))

    boundary_limited = i == len(tp_use) - 1 and np.isclose(
        tp_use[-1], tp[-1], rtol=1e-10, atol=1e-12
    )
    return {
        "time": float(T if boundary_limited else tp_use[i]),
        "resolved": not boundary_limited,
        "peak_time": float(tp_use[i]),
        "peak_value": float(yp_use[i]),
        "objective": float(objective[i]),
    }


def _fit_two_factor_curve(
    code,
    tau,
    y,
    t,
    omega,
    eps,
    skip,
    stride,
    projector_info=None,
):
    """Fit one corrected sensitivity envelope and attach analytic predictions."""
    tp_all, yp_all = _peak_data(t, y, skip=skip, stride=stride)
    if len(tp_all) < 4:
        return None

    if projector_info is None:
        spectral = discrete_x_spectral_decay_rates(code, omega, eps, tau)
        projector_exact = _spectral_projector_rates(code, spectral)
        projector_approx = discrete_x_projector_decay_rate_series(
            code, omega, eps, tau, order=2
        )
    else:
        projector_exact = projector_info["exact"]
        projector_approx = projector_info["tau2"]

    factor_exact = _sensitivity_factors_from_projector_rates(code, projector_exact, tau)
    factor_approx = _sensitivity_factors_from_projector_rates(
        code, projector_approx, tau
    )

    # Fit all selected maxima.  No additional decay-window cutoff is used.
    tp_fit, yp_fit = tp_all, yp_all

    fit = fit_two_factor_envelope(code, tp_fit, yp_fit, tau)
    prediction_all = eval_two_factor_envelope(tp_all, fit)
    log_residual_all = np.log(prediction_all / yp_all)
    relative_all = prediction_all / yp_all - 1.0

    # Choose one protocol stopping time for the fixed design budget
    # T_max=t[-1].  The optimization uses all numerical envelope maxima,
    # without peak thinning, and keeps the integer repetition factor exactly.
    tp_opt, yp_opt = _peak_data(t, y, skip=skip, stride=1)
    T_max = float(t[-1])
    numerical_optimum = _numerical_budget_optimum(tp_opt, yp_opt, T_max)
    t_opt = float(numerical_optimum["time"])

    # Continuous-budget estimate: replace floor(T/t) by T/t and use the exact
    # one-cycle coefficients.  The resulting stationary point is independent
    # of T, but it is capped by the available design budget in the table.
    t_est = _two_factor_optimum_from_factors(
        factor_exact["half"],
        factor_exact["one"],
        tau,
    )

    f_opt = (
        float(numerical_optimum["peak_value"])
        if np.isfinite(numerical_optimum.get("peak_value", np.nan))
        else np.nan
    )
    return {
        "code": code,
        "tau": float(tau),
        "fit": fit,
        "factor_fit": {"half": fit["a_half"], "one": fit["a_one"]},
        "factor_exact": factor_exact,
        "factor_approx": factor_approx,
        "projector_exact": projector_exact,
        "projector_approx": projector_approx,
        "t_peaks": tp_all,
        "y_peaks": yp_all,
        "t_fit": tp_fit,
        "y_fit": yp_fit,
        "t_opt": t_opt,
        "t_opt_resolved": bool(numerical_optimum["resolved"]),
        "t_opt_peak": float(numerical_optimum["peak_time"]),
        "t_est": t_est,
        "f_opt": f_opt,
        "all_log_rmse": float(np.sqrt(np.mean(log_residual_all**2))),
        "all_max_relative_error": float(np.max(np.abs(relative_all))),
    }


def _two_factor_fits_for_code(
    data,
    code,
    tau_values,
    omega,
    eps,
    skip=1,
    stride=2,
):
    """Return one cached fit object per tau, reused by every X-noise figure."""
    cache = data.setdefault("_paper_analysis", {}).setdefault("two_factor_envelope", {})
    key = (
        code.lower(),
        tuple(float(x) for x in tau_values),
        float(omega),
        float(eps),
        int(skip),
        int(stride),
    )
    if key in cache:
        return cache[key]

    t = np.asarray(data["x"][code]["t"], float)
    curves = []
    for tau in tau_values:
        y = np.asarray(data["x"][code]["qec"][float(tau)]["sensitivity"], float)
        projector_info = _ensure_projector_fit(data, code, tau, omega, eps)
        curve = _fit_two_factor_curve(
            code,
            float(tau),
            y,
            t,
            omega,
            eps,
            skip=skip,
            stride=stride,
            projector_info=projector_info,
        )
        if curve is not None:
            curves.append(curve)

    if not curves:
        raise RuntimeError(f"No valid QEC envelope fits for {code}.")

    result = (t, curves)
    cache[key] = result
    return result


def _sigma_crossover(t, curve):
    r"""Fixed-budget protocol performance versus elapsed time t.

    The interrogation duration t_opt is chosen once by optimizing the final
    sensitivity at the fixed design budget T_max=t[-1] used when the curve was
    constructed.  During execution of that protocol, t is the elapsed time.
    Before t_opt the first interrogation is still running; afterwards each run
    lasts t_opt and the number of completed runs is floor(t/t_opt).
    """
    t = np.asarray(t, dtype=float)
    sigma = np.empty_like(t)
    t_opt = float(curve["t_opt"])

    if (
        not bool(curve.get("t_opt_resolved", False))
        or not np.isfinite(t_opt)
        or t_opt <= 0.0
    ):
        return eval_two_factor_envelope(t, curve["fit"])

    before = t <= t_opt
    sigma[before] = eval_two_factor_envelope(t[before], curve["fit"])

    after = ~before
    if np.any(after):
        f_opt_fit = float(eval_two_factor_envelope(t_opt, curve["fit"]))
        repetitions = np.maximum(
            np.floor(t[after] / t_opt + 1e-12),
            1.0,
        )
        sigma[after] = np.sqrt(repetitions) * f_opt_fit

    return sigma


def _sigma_peak_crossover(curve):
    r"""Numerical dots for the same fixed-duration repeated protocol.

    The numerical ``t_opt`` is chosen once for the fixed design budget
    T_max.  Before it, the dots are the directly sampled envelope maxima.
    Afterwards, every interrogation is stopped at that same t_opt, so the
    elapsed-time dependence is the integer number of completed repetitions,

        sqrt(floor(t/t_opt)) * S_num(t_opt).

    This is the numerical analogue of ``_sigma_crossover`` and gives genuine
    plateaus after the crossover.
    """
    t_values = np.asarray(curve["t_peaks"], dtype=float)
    yp = np.asarray(curve["y_peaks"], dtype=float)

    # If the fixed-budget optimum is boundary limited, the turnover has
    # not been resolved inside the simulated interval: all displayed dots are
    # simply the numerical envelope maxima.
    if not bool(curve.get("t_opt_resolved", False)):
        return yp.copy()

    t_opt = float(curve["t_opt"])
    t_opt_peak = float(curve.get("t_opt_peak", t_opt))
    if not np.isfinite(t_opt) or t_opt <= 0.0:
        return yp.copy()

    # Use the exact numerical sensitivity at the fixed-budget optimum.  The
    # optimum itself is computed from all peaks, even if the displayed dots
    # have been thinned for readability.
    y_opt = float(curve.get("f_opt", np.nan))
    if not np.isfinite(y_opt) or y_opt <= 0.0:
        i_opt = int(np.argmin(np.abs(t_values - t_opt_peak)))
        y_opt = float(yp[i_opt])

    sigma = yp.copy()
    after = t_values > t_opt
    if np.any(after):
        repetitions = np.maximum(
            np.floor(t_values[after] / t_opt + 1e-12),
            1.0,
        )
        sigma[after] = np.sqrt(repetitions) * y_opt

    return sigma


def _u_scale(curve):
    # Use exactly the same fixed-budget numerical optimum as in the crossover
    # plot.  The fitted envelope is normalized at that numerical time so the
    # fitted scaled curve passes through (1, 1).
    t_opt = float(curve["t_opt"])
    if not np.isfinite(t_opt) or t_opt <= 0.0:
        raise ValueError(
            f"No positive numerical optimum for "
            f"{curve['code']}, tau={curve['tau']:g}."
        )
    f_scale = float(eval_two_factor_envelope(t_opt, curve["fit"]))
    if not np.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("Invalid fitted sensitivity at numerical t_opt.")
    return t_opt, f_scale


def _sigma_u_crossover(T, curve):
    T = np.asarray(T, dtype=float)
    t_opt, f_scale = _u_scale(curve)
    sigma = np.empty_like(T)
    before = T <= t_opt
    sigma[before] = eval_two_factor_envelope(T[before], curve["fit"])
    sigma[~before] = f_scale * np.sqrt(np.floor(T[~before] / t_opt))
    return sigma


def _sigma_u_scaled(T, curve):
    t_opt, f_scale = _u_scale(curve)
    return np.asarray(T, float) / t_opt, _sigma_u_crossover(T, curve) / f_scale


def _projector_analysis_cache(data):
    return data.setdefault("_paper_analysis", {}).setdefault("projector", {})


def _store_projector_fit(data, code, tau, info):
    cache = _projector_analysis_cache(data)
    cache.setdefault(code, {})[float(tau)] = info


def _ensure_projector_fit(data, code, tau, omega, eps):
    cache = _projector_analysis_cache(data)
    tau = float(tau)
    if code in cache and tau in cache[code]:
        return cache[code][tau]

    d = data["x"][code]
    t = np.asarray(d["t"], float)
    _, _, _, info = fit_projector_qec(
        code, t, d["qec"][tau]["projector"], tau, omega, eps
    )
    _store_projector_fit(data, code, tau, info)
    return info


def _tex_number_body(value, digits=4):
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    if -2 <= exponent <= 2:
        return f"{value:.{digits}g}"
    return rf"{mantissa:.{digits}g}\times10^{{{exponent}}}"


def _tex_number(value, digits=4):
    return rf"${_tex_number_body(value, digits)}$"


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def write_decay_rate_tables(
    data,
    tau_values,
    omega,
    eps,
    skip=1,
    stride=2,
    plot_dir="plots",
):
    """Write projector rates, sensitivity factors, and interrogation optima."""
    os.makedirs(plot_dir, exist_ok=True)

    steane_rows = []
    shor_rows = []
    for tau in tau_values:
        info = _ensure_projector_fit(data, "Steane", tau, omega, eps)
        rmse = info["quality"]["rmse"]
        for key, label in (("real", r"$D_{\mathrm r}$"), ("osc", r"$D_{\mathrm o}$")):
            steane_rows.append(
                " & ".join(
                    [
                        _tex_number(tau, 3),
                        label,
                        _tex_number(info["fit"][key]),
                        _tex_number(info["exact"][key]),
                        _tex_number(info["tau2"][key]),
                        _tex_number(rmse),
                    ]
                )
                + r" \\"
            )

        info = _ensure_projector_fit(data, "Shor", tau, omega, eps)
        shor_rows.append(
            " & ".join(
                [
                    _tex_number(tau, 3),
                    r"$D$",
                    _tex_number(info["fit"]["common"]),
                    _tex_number(info["exact"]["common"]),
                    _tex_number(info["tau2"]["common"]),
                    _tex_number(info["quality"]["rmse"]),
                ]
            )
            + r" \\"
        )

    projector_table = (
        r"""\begin{table}[h]
    \centering
    \caption{Steane projector decay rates. $D_{\mathrm{fit}}$ is obtained from
    the stroboscopic projector, $D_{\mathrm{exact}}$ from the one-cycle
    eigenvalues, and $D_{\mathrm{approx}}$ from the $O(\tau^2)$ expansion.
    RMSE is computed from the fitted projector probability; all decay rates are in $\mathrm{s}^{-1}$.}
    \label{tab:projector-decay-rates-steane}
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        $\tau$ [$\mathrm{s}$] & mode & fit & exact & $O(\tau^2)$ & RMSE \\
        \hline
"""
        + "\n".join("        " + row for row in steane_rows)
        + r"""
        \hline
    \end{tabular}
\end{table}

\begin{table}[h]
    \centering
    \caption{Shor projector decay rates. The notation is the same as in
    Table~\ref{tab:projector-decay-rates-steane}.}
    \label{tab:projector-decay-rates-shor}
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        $\tau$ [$\mathrm{s}$] & mode & fit & exact & $O(\tau^2)$ & RMSE \\
        \hline
"""
        + "\n".join("        " + row for row in shor_rows)
        + r"""
        \hline
    \end{tabular}
\end{table}
"""
    )

    sensitivity_tables = []
    optimum_rows = []
    scale_regime_rows = []

    for code in ("Steane", "Shor"):
        _, curves = _two_factor_fits_for_code(
            data,
            code,
            tau_values,
            omega,
            eps,
            skip=skip,
            stride=stride,
        )

        rows = []
        for curve in curves:
            rmse = curve["fit"]["log_rmse"]
            for key, label in (("half", r"$D_{1/2}$"), ("one", r"$D_1$")):
                rows.append(
                    " & ".join(
                        [
                            _tex_number(curve["tau"], 3),
                            label,
                            _tex_number(curve["factor_fit"][key]),
                            _tex_number(curve["factor_exact"][key]),
                            _tex_number(curve["factor_approx"][key]),
                            _tex_number(rmse),
                        ]
                    )
                    + r" \\"
                )

            T_max = float(data["x"][code]["t"][-1])
            t_num = min(float(curve["t_opt"]), T_max)
            t_approx = (
                min(float(curve["t_est"]), T_max)
                if np.isfinite(curve["t_est"]) and curve["t_est"] > 0.0
                else np.nan
            )
            rel = (
                abs(t_approx - t_num) / t_num
                if np.isfinite(t_approx) and t_num > 0.0
                else np.nan
            )
            continuous_repetitions = T_max / t_num if t_num > 0.0 else np.nan
            integer_repetitions = (
                np.floor(continuous_repetitions + 1e-12)
                if np.isfinite(continuous_repetitions)
                else np.nan
            )
            floor_loss_pct = (
                100.0 * (1.0 - integer_repetitions / continuous_repetitions)
                if np.isfinite(continuous_repetitions) and continuous_repetitions > 0.0
                else np.nan
            )

            optimum_rows.append(
                " & ".join(
                    [
                        code,
                        _tex_number(curve["tau"], 3),
                        _tex_number(t_num),
                        _tex_number(t_approx),
                        _tex_number(rel),
                        _tex_number(floor_loss_pct),
                    ]
                )
                + r" \\"
            )

            scale = _slope_selected_p_scale(curve)
            if scale is not None:
                p_body = r"1/2" if np.isclose(scale["p"], 0.5) else "1"
                scale_regime_rows.append(
                    " & ".join(
                        [
                            code,
                            _tex_number(curve["tau"], 3),
                            _tex_number(scale["ratio_slope"]),
                            rf"${p_body}$",
                            _tex_number(scale["time"]),
                        ]
                    )
                    + r" \\"
                )

        code_lower = code.lower()
        sensitivity_tables.append(
            r"""\begin{table}[h]
    \centering
    \caption{Two-factor envelope coefficients for the """
            + code
            + r""" code. The fitted coefficients are compared with the
    predictions obtained from the exact projector rates and their
    $O(\tau^2)$ approximation. RMSE$_{\log}$ is the root-mean-square residual
    of the logarithmic regression. The units of $D_{1/2}$ and $D_1$ are
    $\mathrm{s}^{-1}$ and $\mathrm{s}^{-2}$, respectively.}
    \label{tab:sensitivity-factors-"""
            + code_lower
            + r"""}
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        $\tau$ [$\mathrm{s}$] & coefficient & fit & exact & $O(\tau^2)$ & RMSE$_{\log}$ \\
        \hline
"""
            + "\n".join("        " + row for row in rows)
            + r"""
        \hline
    \end{tabular}
\end{table}
"""
        )

    sensitivity_table = "\n".join(sensitivity_tables)

    optimum_table = (
        r"""\begin{table}[h]
    \centering
    \caption{Optimal interrogation times for the fixed total simulated budget
    $T_{\max}$. The numerical optimum maximizes the numerical envelope with
    the exact integer repetition factor $\lfloor T_{\max}/t\rfloor$. The
    analytical approximation is obtained by replacing $\lfloor T/t\rfloor$
    by $T/t$ in the two-factor model and is reported as a reference. The
    floor-loss column is
    $100[1-\lfloor T_{\max}/t_{\mathrm{opt}}\rfloor/
    (T_{\max}/t_{\mathrm{opt}})]$. If the turnover is not resolved before
    the end of the simulated interval, the table reports $T_{\max}$.}
    \label{tab:optimal-times}
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
        Code & $\tau$ [$\mathrm{s}$] & $t_{\mathrm{opt}}(T_{\max})$ [$\mathrm{s}$]
        & $\widetilde t_{\mathrm{opt}}(T_{\max},\tau)$ [$\mathrm{s}$]
        & rel. deviation & floor loss [\%] \\
        \hline
"""
        + "\n".join("        " + row for row in optimum_rows)
        + r"""
        \hline
    \end{tabular}
\end{table}
"""
    )

    scale_regime_table = (
        r"""\begin{table}[h]
        \centering
        \caption{Ratio values
        $R_{\mathrm{slope}}$ with corresponding $p$ choices, and associated
        optimal time $t_p$.}
        \label{tab:scale-regimes}
        \begin{tabular}{|c|c|c|c|c|}
            \hline
            Code & $\tau$ [$\mathrm{s}$] & $R_{\mathrm{slope}}$ & $p$ & $t_p$ [$\mathrm{s}$] \\
            \hline
    """
        + "\n".join("        " + row for row in scale_regime_rows)
        + r"""
            \hline
        \end{tabular}
    \end{table}
    """
    )

    paths = {
        "projector": os.path.join(plot_dir, "paper_projector_decay_rates.tex"),
        "sensitivity": os.path.join(plot_dir, "paper_sensitivity_decay_rates.tex"),
        "optimum": os.path.join(plot_dir, "paper_optimal_times.tex"),
        "scale_regime": os.path.join(plot_dir, "paper_scale_regimes.tex"),
    }
    _write_text(paths["projector"], projector_table)
    _write_text(paths["sensitivity"], sensitivity_table)
    _write_text(paths["optimum"], optimum_table)
    _write_text(paths["scale_regime"], scale_regime_table)
    return paths


def z_encoded_inverse_error_envelope(code, t, eps):
    t = np.asarray(t, dtype=float)
    eps = float(eps)

    if code.lower() == "steane":
        r = np.exp(-8.0 * eps * t)
        rad = np.maximum(
            (2.0 - r) * (1.0 - r) * (7.0 * r + 1.0) * (7.0 * r + 2.0),
            0.0,
        )
        den = 2.0 + 9.0 * r - 7.0 * r**2 + np.sqrt(rad)
        return 4.0 * np.sqrt(7.0) * t * r / np.sqrt(den)

    if code.lower() == "shor":
        r = np.exp(-12.0 * eps * t)
        rad = np.maximum((1.0 - r) * (3.0 * r + 1.0), 0.0)
        den = 1.0 + r + np.sqrt(rad)
        return 6.0 * np.sqrt(6.0) * t * r / np.sqrt(den)

    raise ValueError(code)


def z_reference_inverse_error_envelopes(t, eps, n_qubits):
    t = np.asarray(t, dtype=float)
    eps = float(eps)
    n = int(n_qubits)
    product = 2.0 * np.sqrt(n) * t * np.exp(-2.0 * eps * t)
    ghz = 2.0 * n * t * np.exp(-2.0 * n * eps * t)
    return product, ghz


def _common_domain(time_arrays):
    arrays = [np.asarray(x, float) for x in time_arrays if len(x)]
    if not arrays:
        raise ValueError("No non-empty time arrays were supplied.")
    t0 = max(float(x[0]) for x in arrays)
    t1 = min(float(x[-1]) for x in arrays)
    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0 > 0.0):
        raise ValueError(f"Invalid common envelope interval: [{t0}, {t1}].")
    return t0, t1


def _time_grid(t0, t1, n=3000, log_scale=False):
    if log_scale:
        return np.geomspace(t0, t1, n)
    return np.linspace(t0, t1, n)


def _clip_points(t, y, t0, t1):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    keep = (t >= t0) & (t <= t1)
    return t[keep], y[keep]


def _z_panel_title(eps):
    if np.isclose(eps, 0.0):
        return r"$\epsilon=0\,\mathrm{s}^{-1}$"
    return rf"$\epsilon={float(eps):g}\,\mathrm{{s}}^{{-1}}$"


def _plot_projector_z_two_panel(data, z_eps_values, plot_dir):
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharex=True, sharey=True)

    for ax, code in zip(axes, ("Steane", "Shor")):
        for eps_z in z_eps_values:
            d = data["z"][code][float(eps_z)]
            t = np.asarray(d["t"], float)
            p = np.asarray(d["uncorrected"]["projector"], float)
            ax.plot(
                t,
                p,
                color=_z_eps_color(eps_z),
                lw=2.5,
                label=_epsilon_label(eps_z),
            )

        ax.set_title(code)
        ax.set_xlabel(r"Time $t$ [$\mathrm{s}$]")
        ax.set_ylabel(r"Projector probability")
        _decorate(ax)

    shared_legend(fig, axes, ncol=3, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save(fig, plot_dir, "paper_projector_z.pdf")


def plot_projector(
    data,
    noise_type,
    tau_values,
    omega,
    eps,
    z_eps_values=None,
    plot_dir="plots",
):
    if noise_type == "z":
        if z_eps_values is None:
            raise ValueError("z_eps_values must be supplied for Z-noise plots.")
        return _plot_projector_z_two_panel(data, z_eps_values, plot_dir)

    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharex=True, sharey=True)

    for ax, code in zip(axes, ("Steane", "Shor")):
        d = data["x"][code]
        t = np.asarray(d["t"], float)

        for tau in tau_values:
            q = d["qec"][float(tau)]
            ts, ps, fit_fun, fit_info = fit_projector_qec(
                code, t, q["projector"], tau, omega, eps
            )
            _store_projector_fit(data, code, tau, fit_info)

            color = _tau_color(tau)
            step = max(1, len(ts) // 160)
            ax.plot(ts[::step], ps[::step], "o", ms=3.6, alpha=0.20, color=color)
            tf = np.linspace(t[0], t[-1], 4000)
            ax.plot(tf, fit_fun(tf), lw=2.45, color=color, label=_tau_label(tau))

        ax.plot(
            t,
            d["uncorrected"]["projector"],
            color=NO_QEC_COLOR,
            lw=2.3,
            label="No QEC",
        )
        ax.set_title(code)
        ax.set_xlabel(r"Time $t$ [$\mathrm{s}$]")
        ax.set_ylabel(r"Projector probability")
        ax.set_ylim(-0.02, 1.02)
        _decorate(ax)

    shared_legend(fig, axes, ncol=4, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, plot_dir, "paper_projector_x.pdf")


def _plot_sensitivity_z_grid(data, z_eps_values, log_scale, plot_dir):
    paper_style()
    fig, axes = plt.subplots(
        2, len(z_eps_values), figsize=(10.4, 7.2), sharex=True, sharey=False
    )
    axes = np.atleast_2d(axes)

    for row, code in enumerate(("Steane", "Shor")):
        n_qubits = 7 if code == "Steane" else 9
        for col, eps_z in enumerate(z_eps_values):
            ax = axes[row, col]
            d = data["z"][code][float(eps_z)]
            t = np.asarray(d["t"], float)
            grid = t[t > 0.0]
            color = _z_eps_color(eps_z)

            encoded = z_encoded_inverse_error_envelope(code, grid, eps_z)
            product, ghz = z_reference_inverse_error_envelopes(grid, eps_z, n_qubits)

            ax.plot(
                grid,
                encoded,
                color=color,
                lw=2.5,
                ls="-",
                label="Encoded projector",
            )
            ax.plot(
                grid,
                product,
                color=color,
                lw=2.35,
                ls="--",
                label=r"$|+\rangle^{\otimes n}$ probes",
            )
            ax.plot(
                grid,
                ghz,
                color=color,
                lw=2.35,
                ls=":",
                label=r"$|\mathrm{GHZ}_n\rangle$ probe",
            )

            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlim(grid[0], grid[-1])
            if row == 0:
                ax.set_title(_z_panel_title(eps_z))
            if col == 0:
                code_label = (
                    r"\mathrm{St,env}" if code == "Steane" else r"\mathrm{Sh,env}"
                )
                ax.set_ylabel(rf"$\delta\omega^{{-1}}_{{{code_label}}}$ [s]")
            if row == 1:
                ax.set_xlabel(r"Time $t$ [$\mathrm{s}$]")
            _decorate(ax)

    handles = [
        Line2D([], [], color=REFERENCE_GRAY, lw=2.5, ls="-", label="Encoded projector"),
        Line2D(
            [],
            [],
            color=REFERENCE_GRAY,
            lw=2.35,
            ls="--",
            label=r"$|+\rangle^{\otimes n}$ probes",
        ),
        Line2D(
            [],
            [],
            color=REFERENCE_GRAY,
            lw=2.35,
            ls=":",
            label=r"$|\mathrm{GHZ}_n\rangle$ probe",
        ),
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, plot_dir, "paper_inverse_error_envelope_z.pdf")


def plot_sensitivity_envelope(
    data,
    noise_type,
    tau_values,
    eps,
    z_eps_values=None,
    skip=1,
    stride=2,
    log_scale=True,
    omega=None,
    plot_dir="plots",
):
    """Plot the exact maxima and the cached two-factor fit evaluated directly."""
    if noise_type == "z":
        if z_eps_values is None:
            raise ValueError("z_eps_values must be supplied for Z-noise plots.")
        return _plot_sensitivity_z_grid(data, z_eps_values, log_scale, plot_dir)

    if omega is None:
        raise ValueError("omega must be supplied for the X-noise fit.")

    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharex=True)

    for ax, code in zip(axes, ("Steane", "Shor")):
        d = data["x"][code]
        t = np.asarray(d["t"], float)
        _, curves = _two_factor_fits_for_code(
            data,
            code,
            tau_values,
            omega,
            eps,
            skip=skip,
            stride=stride,
        )

        reference_curves = []
        tp_no, yp_no = _peak_data(
            t, d["uncorrected"]["sensitivity"], skip=skip, stride=1
        )
        if len(tp_no) >= 2:
            reference_curves.append(("No QEC", NO_QEC_COLOR, "-", 2.0, tp_no, yp_no))

        ref_specs = (
            (
                "product_sensitivity",
                r"$|+\rangle^{\otimes n}$ probes",
                REFERENCE_GRAY,
                "--",
            ),
            (
                "ghz_sensitivity",
                r"$|\mathrm{GHZ}_n\rangle$ probe",
                GHZ_COLOR,
                ":",
            ),
        )
        for key, label, color, ls in ref_specs:
            tp_ref, yp_ref = _peak_data(t, d["references"][key], skip=skip, stride=1)
            if len(tp_ref) >= 2:
                reference_curves.append((label, color, ls, 2.2, tp_ref, yp_ref))

        time_arrays = [curve["t_peaks"] for curve in curves]
        time_arrays += [reference[4] for reference in reference_curves]
        t0, t1 = _common_domain(time_arrays)
        grid = _time_grid(t0, t1, n=3200, log_scale=log_scale)

        for curve in curves:
            color = _tau_color(curve["tau"])
            ax.plot(
                grid,
                eval_two_factor_envelope(grid, curve["fit"]),
                color=color,
                lw=2.25,
                ls="-",
                label=_tau_label(curve["tau"]),
            )
            tx, yx = _clip_points(curve["t_peaks"], curve["y_peaks"], t0, t1)
            ax.plot(tx, yx, "o", ms=3.8, alpha=0.22, color=color)

        for label, color, ls, lw, tp_ref, yp_ref in reference_curves:
            ax.plot(
                grid,
                np.interp(grid, tp_ref, yp_ref),
                color=color,
                lw=lw,
                ls=ls,
                label=label,
            )

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-2)
        ax.set_xlim(t0, t1)
        ax.set_title(code)
        ax.set_xlabel(r"Time $t$ [$\mathrm{s}$]")
        ax.set_ylabel(r"$(\delta\omega^{-1})_{\mathrm{env}}$ [$\mathrm{s}$]")
        _decorate(ax)

    shared_legend(fig, axes, ncol=3, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save(fig, plot_dir, "paper_inverse_error_envelope_x.pdf")


def plot_hl_sql_crossover(
    data,
    tau_values,
    omega,
    eps,
    skipped_peaks=1,
    peak_stride=2,
    log_scale=True,
    plot_dir="plots",
):
    """Fixed-budget protocol crossover versus elapsed time t.

    First choose t_opt by maximizing the final floor-aware sensitivity for the
    fixed design budget T_max=t[-1].  Then keep that interrogation duration
    fixed and show how the protocol sensitivity accumulates as the elapsed
    time t runs from the beginning of the experiment up to T_max.
    """
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharex=True)

    for ax, code in zip(axes, ("Steane", "Shor")):
        t, curves = _two_factor_fits_for_code(
            data,
            code,
            tau_values,
            omega,
            eps,
            skip=skipped_peaks,
            stride=peak_stride,
        )
        t0 = max(float(curve["t_peaks"][0]) for curve in curves)
        t_max = float(t[-1])

        for curve in curves:
            color = _tau_color(curve["tau"])
            t_opt = float(curve["t_opt"])
            resolved = bool(curve.get("t_opt_resolved", False))

            # Include the fixed protocol stopping time and every integer-repeat
            # threshold explicitly so that the staircase remains visible.
            t_plot = _time_grid(t0, t_max, n=5000, log_scale=log_scale)
            if resolved and np.isfinite(t_opt) and t_opt > 0.0 and t_opt < t_max:
                n_rep_max = int(np.floor(t_max / t_opt + 1e-12))
                thresholds = t_opt * np.arange(1, n_rep_max + 1, dtype=float)
                thresholds = thresholds[(thresholds >= t0) & (thresholds <= t_max)]
                t_plot = np.unique(
                    np.concatenate((t_plot, thresholds, np.array([t_opt])))
                )

            sigma = _sigma_crossover(t_plot, curve)
            if resolved and np.isfinite(t_opt) and t_opt > 0.0 and t_opt < t_max:
                before = t_plot <= t_opt
                after = t_plot >= t_opt
                ax.plot(
                    t_plot[before],
                    sigma[before],
                    color=color,
                    lw=2.5,
                    label=_tau_label(curve["tau"]),
                )
                ax.step(
                    t_plot[after],
                    sigma[after],
                    where="post",
                    color=color,
                    lw=2.5,
                    label="_nolegend_",
                )
            else:
                ax.plot(
                    t_plot,
                    sigma,
                    color=color,
                    lw=2.5,
                    label=_tau_label(curve["tau"]),
                )

            sigma_p = _sigma_peak_crossover(curve)
            tp, sp = _clip_points(curve["t_peaks"], sigma_p, t0, t_max)
            ax.plot(tp, sp, "o", color=color, ms=5.5, alpha=0.42)

            # Mark the stopping time chosen for the fixed design budget T_max.
            t_mark = float(curve["t_opt"])
            if resolved and np.isfinite(t_mark) and t0 <= t_mark <= t_max:
                sigma_mark = float(_sigma_crossover(np.array([t_mark]), curve)[0])
                ax.plot(
                    t_mark,
                    sigma_mark,
                    "*",
                    color=color,
                    ms=12,
                    mec=MASTER_COLOR,
                    mew=0.45,
                    label="_nolegend_",
                )

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(t0, t_max)
        ax.set_title(code)
        ax.set_xlabel(r"Elapsed time $t$ [$\mathrm{s}$]")
        ax.set_ylabel(r"$\Sigma_{\tau}(t;T)$ [$\mathrm{s}$]")
        _decorate(ax)

    shared_legend(fig, axes, ncol=3, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, plot_dir, "paper_HL_SQL_crossover_x.pdf")


def _code_tau_label(code, tau):
    prefix = "St" if code.lower() == "steane" else "Sh"
    return rf"$\tau_{{\mathrm{{{prefix}}}}}={float(tau):g}\,\mathrm{{s}}$"


def _slope_selected_p_scale(curve):
    r"""Select p from the slope contributions at the numerical t_opt.

    The fitted envelope is

        S(t) = C t exp[-D_{1/2} sqrt(tau t) - D_1 tau t].

    For repeated interrogation, the continuous objective is S(t)/sqrt(t).
    Its logarithmic slope is

        d log[S(t)/sqrt(t)] / d log(t)
            = 1/2 - (1/2) D_{1/2} sqrt(tau t) - D_1 tau t.

    Therefore the square-root and linear decay terms contribute to the slope
    with weights

        Q_{1/2}(t) = (1/2) D_{1/2} sqrt(tau t),
        Q_1(t)     = D_1 tau t.

    At the fixed-budget numerical optimum, define

        R_slope = Q_1(t_opt) / Q_{1/2}(t_opt)
                = 2 D_1 tau t_opt /
                  [D_{1/2} sqrt(tau t_opt)].

    We select p=1/2 when R_slope<1 and p=1 when R_slope>=1.  The associated
    one-term scale is

        t_{1/2} = 1 / (D_{1/2}^2 tau),
        t_1     = 1 / (2 D_1 tau).
    """
    fit = curve["fit"]
    D_half = float(fit["a_half"])
    D_one = float(fit["a_one"])
    tau = float(fit["tau"])
    t_opt = float(curve["t_opt"])

    if (
        not np.isfinite(D_half)
        or not np.isfinite(D_one)
        or not np.isfinite(tau)
        or not np.isfinite(t_opt)
        or D_half < 0.0
        or D_one < 0.0
        or tau <= 0.0
        or t_opt <= 0.0
    ):
        return None

    half_exponent = D_half * np.sqrt(tau * t_opt)
    one_exponent = D_one * tau * t_opt
    half_slope = 0.5 * half_exponent
    one_slope = one_exponent

    if half_slope > 0.0:
        ratio_slope = one_slope / half_slope
    elif one_slope > 0.0:
        ratio_slope = np.inf
    else:
        ratio_slope = 0.0

    if ratio_slope < 1.0 and D_half > 0.0:
        p = 0.5
        t_p = 1.0 / (D_half * D_half * tau)
        regime = "sqrt"
    elif D_one > 0.0:
        p = 1.0
        t_p = 1.0 / (2.0 * D_one * tau)
        regime = "linear"
    elif D_half > 0.0:
        p = 0.5
        t_p = 1.0 / (D_half * D_half * tau)
        regime = "sqrt"
    else:
        return None

    if not np.isfinite(t_p) or t_p <= 0.0:
        return None

    f_p = float(eval_two_factor_envelope(t_p, fit))
    if not np.isfinite(f_p) or f_p <= 0.0:
        return None

    return {
        "p": float(p),
        "time": float(t_p),
        "value": float(f_p),
        "regime": regime,
        "half_exponent": float(half_exponent),
        "one_exponent": float(one_exponent),
        "half_slope": float(half_slope),
        "one_slope": float(one_slope),
        "ratio_slope": float(ratio_slope),
    }


def _p_master_continuous(z):
    r"""Dominant-term master used for the jump-divided p-collapse.

    For either p=1/2 or p=1, the one-term envelope gives

        Sigma_0(z) = z exp[(1-z)/2],     z <= 1,
        Sigma_0(z) = 1,                  z > 1,

    after dividing the SQL branch by the jump factor J(T;t_p).
    """
    z = np.asarray(z, dtype=float)
    y = np.ones_like(z)
    before = z <= 1.0
    y[before] = z[before] * np.exp((1.0 - z[before]) / 2.0)
    return y


def _code_tau_p_label(code, tau, p):
    prefix = "St" if code.lower() == "steane" else "Sh"
    p_text = r"\frac{1}{2}" if np.isclose(p, 0.5) else "1"
    return (
        rf"$\tau_{{\mathrm{{{prefix}}}}}={float(tau):g}\,\mathrm{{s}},"
        rf"\ p={p_text}$"
    )


def _sigma_crossover_at_stop(T, curve, t_stop):
    r"""HL-to-SQL crossover using an explicitly supplied stopping time.

    This is the same construction as ``_sigma_crossover`` but the change of
    behaviour occurs at ``t_stop`` rather than at the numerical t_opt.
    """
    T = np.asarray(T, dtype=float)
    t_stop = float(t_stop)
    if not np.isfinite(t_stop) or t_stop <= 0.0:
        return eval_two_factor_envelope(T, curve["fit"])

    sigma = np.empty_like(T)
    before = T <= t_stop
    sigma[before] = eval_two_factor_envelope(T[before], curve["fit"])

    after = ~before
    if np.any(after):
        f_stop = float(eval_two_factor_envelope(t_stop, curve["fit"]))
        repetitions = np.maximum(
            np.floor(T[after] / t_stop + 1e-12),
            1.0,
        )
        sigma[after] = np.sqrt(repetitions) * f_stop
    return sigma


def _sigma_peak_crossover_at_stop(curve, t_stop):
    r"""Numerical envelope dots for a crossover imposed at ``t_stop``."""
    t_values = np.asarray(curve["t_peaks"], dtype=float)
    y_values = np.asarray(curve["y_peaks"], dtype=float)
    t_stop = float(t_stop)

    if not np.isfinite(t_stop) or t_stop <= 0.0 or t_stop >= t_values[-1]:
        return y_values.copy()

    # Interpolate the sampled numerical envelope at the imposed stopping time.
    # The universal-curve figure is a diagnostic construction, so t_stop need not be
    # one of the original numerical peak times.
    y_stop = float(np.interp(t_stop, t_values, y_values))
    sigma = y_values.copy()
    after = t_values > t_stop
    if np.any(after):
        repetitions = np.maximum(
            np.floor(t_values[after] / t_stop + 1e-12),
            1.0,
        )
        sigma[after] = np.sqrt(repetitions) * y_stop
    return sigma


def _jump_factor(T, t_stop):
    r"""Return the staircase factor used in the SQL branch.

    This equals 1 before the imposed crossover time t_stop and
    sqrt(floor(T/t_stop)) afterwards. Dividing by this factor removes the
    step heights from the universal-curve crossover plot while leaving the location
    of the change of behaviour at T=t_p unchanged.
    """
    T = np.asarray(T, dtype=float)
    t_stop = float(t_stop)
    if not np.isfinite(t_stop) or t_stop <= 0.0:
        return np.ones_like(T)

    factor = np.ones_like(T)
    after = T > t_stop
    if np.any(after):
        factor[after] = np.sqrt(np.maximum(np.floor(T[after] / t_stop + 1e-12), 1.0))
    return factor


def plot_universal_curve_p(
    data,
    tau_values,
    omega,
    eps,
    skipped_peaks=1,
    peak_stride=2,
    log_scale=True,
    plot_dir="plots",
):
    r"""Slope-selected p-scaling with the crossover imposed at t_p.

    The fitted envelope contains square-root and linear decay terms. We choose
    between p=1/2 and p=1 by comparing their contributions to the logarithmic
    slope of the repeated-interrogation objective S(t)/sqrt(t) at the
    fixed-budget numerical t_opt. The ratio is

        R_slope = 2 D_1 tau t_opt /
                  [D_{1/2} sqrt(tau t_opt)].

    We use p=1/2 when R_slope<1 and p=1 when R_slope>=1. The corresponding
    one-term scale is t_p. The plotted crossover is Sigma_tau(T;t_p).

    To remove the SQL staircase heights, divide by

        J(T;t_p) = 1,                                  T <= t_p,
                   sqrt(floor(T/t_p)),                  T > t_p.

    With

        z = (T/t_p)^p,
        Sigma_{tau,p}^U(z)
            = [Sigma_tau(T;t_p)/(delta_omega_fit^{-1}(t_p) J(T;t_p))]^p,

    the one-term reference is Sigma_0(z)=z exp[(1-z)/2] for z<=1 and 1 for
    z>1.
    """
    paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    plotted = []

    for code in ("Steane", "Shor"):
        t, curves = _two_factor_fits_for_code(
            data,
            code,
            tau_values,
            omega,
            eps,
            skip=skipped_peaks,
            stride=peak_stride,
        )
        t_max = float(t[-1])

        for curve in curves:
            scale = _slope_selected_p_scale(curve)
            if scale is None:
                continue

            p_exp = float(scale["p"])
            t_p = float(scale["time"])
            f_p = float(scale["value"])

            T_grid = _time_grid(
                float(curve["t_peaks"][0]), t_max, n=5000, log_scale=log_scale
            )

            # Preserve every integer-repeat threshold generated by t_p.
            if t_p < t_max:
                n_max = int(np.floor(t_max / t_p + 1e-12))
                thresholds = t_p * np.arange(1, n_max + 1, dtype=float)
                thresholds = thresholds[
                    (thresholds >= T_grid[0]) & (thresholds <= t_max)
                ]
                T_grid = np.unique(
                    np.concatenate((T_grid, thresholds, np.array([t_p])))
                )

            z = (T_grid / t_p) ** p_exp
            sigma = _sigma_crossover_at_stop(T_grid, curve, t_p)
            jump = _jump_factor(T_grid, t_p)
            y = (sigma / (f_p * jump)) ** p_exp

            T_points = np.asarray(curve["t_peaks"], float)
            zp = (T_points / t_p) ** p_exp
            sigma_p = _sigma_peak_crossover_at_stop(curve, t_p)
            jump_p = _jump_factor(T_points, t_p)
            yp = (sigma_p / (f_p * jump_p)) ** p_exp

            good = np.isfinite(z) & np.isfinite(y) & (z > 0.0) & (y > 0.0)
            z, y = z[good], y[good]
            goodp = np.isfinite(zp) & np.isfinite(yp) & (zp > 0.0) & (yp > 0.0)
            zp, yp = zp[goodp], yp[goodp]
            if not len(z):
                continue

            if code == "Steane":
                color = UNIVERSAL_TAU_COLORS_ST[float(curve["tau"])]
            else:
                color = UNIVERSAL_TAU_COLORS_SH[float(curve["tau"])]

            plotted.append((code, curve, z, y, zp, yp, color, p_exp))

    if not plotted:
        raise RuntimeError("No valid universal-curve crossover data were available.")

    zmin = min(float(z[0]) for _, _, z, _, _, _, _, _ in plotted)
    zmax = max(float(z[-1]) for _, _, z, _, _, _, _, _ in plotted)
    zmin_plot = max(0.9 * zmin, 1e-4) if log_scale else max(0.0, 0.9 * zmin)
    zmax_plot = min(10.0, 1.05 * zmax)
    zmax_plot = max(zmax_plot, 2.05)

    handles = []
    labels = []
    for code, curve, z, y, zp, yp, color, p_exp in plotted:
        keep = (z >= zmin_plot) & (z <= zmax_plot)
        zk, yk = z[keep], y[keep]
        if not len(zk):
            continue

        ls = "-" if np.isclose(p_exp, 0.5) else "--"
        before = zk <= 1.0
        after = zk >= 1.0
        if np.any(before):
            (line,) = ax.plot(
                zk[before],
                yk[before],
                color=color,
                lw=2.35,
                ls=ls,
            )
        else:
            (line,) = ax.plot([], [], color=color, lw=2.35, ls=ls)

        if np.any(after):
            ax.step(
                zk[after],
                yk[after],
                where="post",
                color=color,
                lw=2.35,
                ls=ls,
            )

        point_keep = (zp >= zmin_plot) & (zp <= zmax_plot)
        ax.plot(
            zp[point_keep],
            yp[point_keep],
            "o",
            color=color,
            ms=3.8,
            alpha=0.22,
        )

        handles.append(line)
        labels.append(_code_tau_p_label(code, curve["tau"], p_exp))

    zr = (
        np.geomspace(zmin_plot, zmax_plot, 5000)
        if log_scale
        else np.linspace(zmin_plot, zmax_plot, 5000)
    )
    yr = np.ones_like(zr)
    before_ref = zr <= 1.0
    after_ref = zr >= 1.0
    (ref_line,) = ax.plot(
        zr[before_ref],
        _p_master_continuous(zr[before_ref]),
        color=MASTER_COLOR,
        ls=":",
        lw=1.35,
        alpha=0.42,
    )
    if np.any(after_ref):
        ax.plot(
            zr[after_ref],
            yr[after_ref],
            color=MASTER_COLOR,
            ls=":",
            lw=1.35,
            alpha=0.42,
        )
    handles.append(ref_line)
    labels.append(r"$\Sigma_{0}^{\mathrm{U}}(z)$")

    tp_line = ax.axvline(1.0, color=CROSSOVER_COLOR, ls="-.", lw=1.0, alpha=0.75)
    handles.append(tp_line)
    labels.append(r"$z=1$")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlim(zmin_plot, zmax_plot)
    ax.set_xlabel(r"Scaled budget $z=(T/t_p)^p$")
    ax.set_ylabel(r"$\Sigma_{\tau,p}^{\mathrm{U}}(z)$")
    _decorate(ax)

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.74)
    axes_box = ax.get_position()
    legend_center = 0.5 * (axes_box.x0 + axes_box.x1)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(legend_center, 0.985),
        ncol=3,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
    )
    return _save(fig, plot_dir, "paper_universal_curve_p_x.pdf")


def plot_universal_curve(
    data,
    tau_values,
    omega,
    eps,
    skipped_peaks=1,
    peak_stride=2,
    log_scale=True,
    plot_dir="plots",
):
    """Universal-curve normalization using the same fixed-budget numerical t_opt."""
    paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    plotted = []

    for code in ("Steane", "Shor"):
        t, curves = _two_factor_fits_for_code(
            data,
            code,
            tau_values,
            omega,
            eps,
            skip=skipped_peaks,
            stride=peak_stride,
        )
        t_max = float(t[-1])

        for curve in curves:
            t_opt = float(curve["t_opt"])
            if not np.isfinite(t_opt) or t_opt <= 0.0:
                continue
            # Keep all requested tau values and use exactly the same fixed-budget
            # numerical optimum as in the HL-to-SQL crossover plot.

            T_grid = _time_grid(
                float(curve["t_peaks"][0]), t_max, n=5000, log_scale=log_scale
            )
            x, sigma_u = _sigma_u_scaled(T_grid, curve)
            good = np.isfinite(x) & np.isfinite(sigma_u) & (x > 0.0) & (sigma_u > 0.0)
            x, sigma_u = x[good], sigma_u[good]
            if not len(x):
                continue

            if code == "Steane":
                color = UNIVERSAL_TAU_COLORS_ST[float(curve["tau"])]
            else:
                color = UNIVERSAL_TAU_COLORS_SH[float(curve["tau"])]
            plotted.append((code, curve, x, sigma_u, color))

    if not plotted:
        raise RuntimeError("No valid universal-curve plots were available.")

    xmin = min(float(x[0]) for _, _, x, _, _ in plotted)
    xmax = max(float(x[-1]) for _, _, x, _, _ in plotted)
    xmin_plot = max(0.9 * xmin, 1e-3) if log_scale else max(0.0, 0.9 * xmin)
    xmax_plot = min(8.0, 1.05 * xmax)
    xmax_plot = max(xmax_plot, 2.05)

    handles = []
    labels = []
    for code, curve, x, sigma_u, color in plotted:
        keep = (x >= xmin_plot) & (x <= xmax_plot)
        (line,) = ax.plot(
            x[keep],
            sigma_u[keep],
            color=color,
            lw=2.35,
            ls="-",
        )
        handles.append(line)
        labels.append(_code_tau_label(code, curve["tau"]))

    # x=1 marks the same fixed-budget numerical optimum used in the
    # crossover figure.  No continuous no-floor reference is drawn here.
    ax.axvline(1.0, color=MASTER_COLOR, ls=":", lw=0.9, alpha=0.22)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_xlabel(r"Scaled budget $x=T/t_{\mathrm{opt}}$")
    ax.set_ylabel(r"$\Sigma_{\tau}^{\mathrm{U}}(x)$")
    _decorate(ax)

    # Reserve a band above the axes, then center the legend over the actual
    # plotting area rather than over the full figure canvas.
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.76)
    axes_box = ax.get_position()
    legend_center = 0.5 * (axes_box.x0 + axes_box.x1)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(legend_center, 0.98),
        ncol=4,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
    )
    primary_path = _save(fig, plot_dir, "paper_universal_curve_x.pdf")
    plot_universal_curve_p(
        data,
        tau_values,
        omega,
        eps,
        skipped_peaks=skipped_peaks,
        peak_stride=peak_stride,
        log_scale=log_scale,
        plot_dir=plot_dir,
    )
    return primary_path
