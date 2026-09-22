"""Generate all figures and tables used by the numerical-analysis section."""

from pathlib import Path
import pickle

import numpy as np

from analysis import compute_all_analytics, corrected_projector_and_sensitivity
from plotting import (
    plot_hl_sql_crossover,
    plot_projector,
    plot_sensitivity_envelope,
    plot_universal_curve,
    write_decay_rate_tables,
)

OMEGA = 3.0
X_NOISE_RATE = 0.1
X_MAX_TIME = 40.0
Z_MAX_TIME = 20.0
NUM_POINTS = 12_000

CORRECTION_INTERVALS = (1e-3, 5e-2, 1e-1)
Z_NOISE_RATES = (1e-3, 1e-2, 1e-1)
CODES = ("Steane", "Shor")

SKIPPED_PEAKS = 1
PEAK_STRIDE = 2
LOG_SCALE = True
FORCE_RERUN = False

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "sim_data"
PLOT_DIR = ROOT / "plots"
CACHE_PATH = DATA_DIR / "paper_curves.pkl"


def _parameters():
    return {
        "omega": OMEGA,
        "x_noise_rate": X_NOISE_RATE,
        "x_max_time": X_MAX_TIME,
        "z_max_time": Z_MAX_TIME,
        "num_points": NUM_POINTS,
        "correction_intervals": CORRECTION_INTERVALS,
        "z_noise_rates": Z_NOISE_RATES,
        "codes": CODES,
    }


def _uncorrected_curves(code, noise_type, times, noise_rate):
    result = compute_all_analytics(
        OMEGA,
        noise_rate,
        times,
        code,
        noise_type=noise_type,
    )
    encoded = {
        "projector": result["exp_projector"],
        "sensitivity": result["sens_projector"],
    }
    references = {
        "product_probability": result["product_probability"],
        "product_sensitivity": result["product_sensitivity"],
        "ghz_probability": result["ghz_probability"],
        "ghz_sensitivity": result["ghz_sensitivity"],
    }
    return encoded, references


def collect_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists() and not FORCE_RERUN:
        try:
            package = pickle.loads(CACHE_PATH.read_bytes())
            if package.get("parameters") == _parameters():
                return package["data"]
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
            pass

    data = {"x": {}, "z": {}}

    x_times = np.linspace(0.0, X_MAX_TIME, NUM_POINTS)
    for code in CODES:
        uncorrected, references = _uncorrected_curves(
            code,
            "x",
            x_times,
            X_NOISE_RATE,
        )
        code_data = {
            "t": x_times,
            "uncorrected": uncorrected,
            "references": references,
            "qec": {},
        }
        for tau in CORRECTION_INTERVALS:
            projector, sensitivity = corrected_projector_and_sensitivity(
                code,
                x_times,
                OMEGA,
                X_NOISE_RATE,
                tau,
            )
            code_data["qec"][float(tau)] = {
                "projector": projector,
                "sensitivity": sensitivity,
            }
        data["x"][code] = code_data

    z_times = np.linspace(0.0, Z_MAX_TIME, NUM_POINTS)
    for code in CODES:
        data["z"][code] = {}
        for noise_rate in Z_NOISE_RATES:
            uncorrected, references = _uncorrected_curves(
                code,
                "z",
                z_times,
                noise_rate,
            )
            data["z"][code][float(noise_rate)] = {
                "t": z_times,
                "uncorrected": uncorrected,
                "references": references,
            }

    CACHE_PATH.write_bytes(
        pickle.dumps({"parameters": _parameters(), "data": data})
    )
    return data


def generate_figures(data):
    plot_projector(
        data,
        "z",
        CORRECTION_INTERVALS,
        OMEGA,
        X_NOISE_RATE,
        z_eps_values=Z_NOISE_RATES,
        plot_dir=PLOT_DIR,
    )
    plot_sensitivity_envelope(
        data,
        "z",
        CORRECTION_INTERVALS,
        X_NOISE_RATE,
        z_eps_values=Z_NOISE_RATES,
        skip=SKIPPED_PEAKS,
        stride=PEAK_STRIDE,
        log_scale=LOG_SCALE,
        plot_dir=PLOT_DIR,
    )
    plot_projector(
        data,
        "x",
        CORRECTION_INTERVALS,
        OMEGA,
        X_NOISE_RATE,
        plot_dir=PLOT_DIR,
    )
    plot_sensitivity_envelope(
        data,
        "x",
        CORRECTION_INTERVALS,
        X_NOISE_RATE,
        skip=SKIPPED_PEAKS,
        stride=PEAK_STRIDE,
        log_scale=LOG_SCALE,
        omega=OMEGA,
        plot_dir=PLOT_DIR,
    )
    plot_hl_sql_crossover(
        data,
        CORRECTION_INTERVALS,
        OMEGA,
        X_NOISE_RATE,
        skipped_peaks=SKIPPED_PEAKS,
        peak_stride=PEAK_STRIDE,
        log_scale=LOG_SCALE,
        plot_dir=PLOT_DIR,
    )
    plot_universal_curve(
        data,
        CORRECTION_INTERVALS,
        OMEGA,
        X_NOISE_RATE,
        skipped_peaks=SKIPPED_PEAKS,
        peak_stride=PEAK_STRIDE,
        log_scale=LOG_SCALE,
        plot_dir=PLOT_DIR,
    )
    write_decay_rate_tables(
        data,
        CORRECTION_INTERVALS,
        OMEGA,
        X_NOISE_RATE,
        skip=SKIPPED_PEAKS,
        stride=PEAK_STRIDE,
        plot_dir=PLOT_DIR,
    )


def main():
    generate_figures(collect_data())


if __name__ == "__main__":
    main()
