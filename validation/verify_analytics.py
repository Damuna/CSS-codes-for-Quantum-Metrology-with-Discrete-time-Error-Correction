"""Compare the paper analytics with an independent full-state simulation.

Run from the repository root with

    python -m validation.verify_analytics

The default suite checks uncorrected X/Z dynamics and one representative
periodically corrected X-noise case for both Steane and Shor codes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from analysis import compute_all_analytics, corrected_projector_and_sensitivity
from .numerical_simulation import simulate_projector


@dataclass(frozen=True)
class Comparison:
    code: str
    case: str
    probability_rmse: float
    probability_max_error: float
    sensitivity_rmse: float
    sensitivity_max_error: float


def _error_metrics(numerical, analytical):
    numerical = np.asarray(numerical, dtype=float)
    analytical = np.asarray(analytical, dtype=float)
    difference = numerical - analytical
    return float(np.sqrt(np.mean(difference**2))), float(np.max(np.abs(difference)))


def _sample_times(t_max, dt, n_samples):
    max_step = int(round(float(t_max) / float(dt)))
    if not np.isclose(max_step * dt, t_max, rtol=0.0, atol=1e-12):
        raise ValueError("t_max must be an integer multiple of dt.")
    sample_steps = np.unique(
        np.rint(np.linspace(0, max_step, int(n_samples))).astype(int)
    )
    return sample_steps * float(dt)


def compare_uncorrected(code, noise_type, omega, epsilon, times, dt, domega):
    numerical = simulate_projector(
        code,
        omega,
        epsilon,
        times,
        noise_type=noise_type,
        dt=dt,
        finite_difference_step=domega,
    )
    analytical = compute_all_analytics(
        omega, epsilon, times, code, noise_type=noise_type
    )

    p_rmse, p_max = _error_metrics(
        numerical["probability"], analytical["exp_projector"]
    )
    s_rmse, s_max = _error_metrics(
        numerical["sensitivity"], analytical["sens_projector"]
    )
    return Comparison(
        code=code,
        case=f"uncorrected {noise_type.upper()}",
        probability_rmse=p_rmse,
        probability_max_error=p_max,
        sensitivity_rmse=s_rmse,
        sensitivity_max_error=s_max,
    )


def compare_corrected_x(code, omega, epsilon, tau, times, dt, domega):
    numerical = simulate_projector(
        code,
        omega,
        epsilon,
        times,
        noise_type="x",
        correction_interval=tau,
        dt=dt,
        finite_difference_step=domega,
    )
    probability, sensitivity = corrected_projector_and_sensitivity(
        code, times, omega, epsilon, tau
    )

    p_rmse, p_max = _error_metrics(numerical["probability"], probability)
    s_rmse, s_max = _error_metrics(numerical["sensitivity"], sensitivity)
    return Comparison(
        code=code,
        case=f"corrected X, tau={tau:g}",
        probability_rmse=p_rmse,
        probability_max_error=p_max,
        sensitivity_rmse=s_rmse,
        sensitivity_max_error=s_max,
    )


def _print_results(results):
    header = (
        f"{'Code':<8} {'Case':<26} "
        f"{'P RMSE':>11} {'P max':>11} {'S RMSE':>11} {'S max':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.code:<8} {result.case:<26} "
            f"{result.probability_rmse:11.3e} "
            f"{result.probability_max_error:11.3e} "
            f"{result.sensitivity_rmse:11.3e} "
            f"{result.sensitivity_max_error:11.3e}"
        )


def run_suite(
    *,
    omega=3.0,
    epsilon=0.1,
    tau=0.05,
    t_max=0.20,
    dt=0.005,
    domega=1e-4,
    n_samples=21,
):
    """Run the standard independent validation suite and return its metrics."""
    times = _sample_times(t_max, dt, n_samples)
    results = []
    for code in ("Steane", "Shor"):
        results.append(
            compare_uncorrected(code, "x", omega, epsilon, times, dt, domega)
        )
        results.append(
            compare_uncorrected(code, "z", omega, epsilon, times, dt, domega)
        )
        results.append(
            compare_corrected_x(code, omega, epsilon, tau, times, dt, domega)
        )
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=0.20)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--domega", type=float, default=1e-4)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument(
        "--max-probability-error",
        type=float,
        default=None,
        help="Exit nonzero if any maximum probability error exceeds this value.",
    )
    parser.add_argument(
        "--max-sensitivity-error",
        type=float,
        default=None,
        help="Exit nonzero if any maximum sensitivity error exceeds this value.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    results = run_suite(
        omega=args.omega,
        epsilon=args.epsilon,
        tau=args.tau,
        t_max=args.t_max,
        dt=args.dt,
        domega=args.domega,
        n_samples=args.samples,
    )
    _print_results(results)

    failed = False
    if args.max_probability_error is not None:
        failed |= any(
            result.probability_max_error > args.max_probability_error
            for result in results
        )
    if args.max_sensitivity_error is not None:
        failed |= any(
            result.sensitivity_max_error > args.max_sensitivity_error
            for result in results
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
