"""Independent full-state numerical simulation for validating paper analytics.

The implementation evolves the density matrix in Liouville space using an
explicit second-order Runge-Kutta step.  It intentionally does not reuse the
reduced-basis evolution implemented in :mod:`analysis`.
"""

from __future__ import annotations

import numpy as np
import qutip as qt
import scipy.sparse as sp

from .quantum_codes import build_correction_superoperator, get_qubit_setup


def as_scipy_csr(obj):
    """Convert a QuTiP or SciPy matrix-like object to SciPy CSR format."""
    if isinstance(obj, qt.Qobj):
        obj = obj.data
    if hasattr(obj, "as_scipy"):
        return obj.as_scipy().tocsr()
    if hasattr(obj, "as_ndarray"):
        return sp.csr_matrix(obj.as_ndarray())
    if sp.issparse(obj):
        return obj.tocsr()
    return sp.csr_matrix(obj)


def _expectation(row_operator, state_vector):
    return float(np.real(row_operator.dot(state_vector)[0, 0]))


def _liouvillian(code, n_qubits, omega, epsilon, noise_type):
    z_terms = []
    collapse_operators = []

    for qubit in range(n_qubits):
        factors = [qt.qeye(2)] * n_qubits
        factors[qubit] = qt.sigmaz()
        z_terms.append(qt.tensor(factors))

        if epsilon > 0.0:
            noise_factors = [qt.qeye(2)] * n_qubits
            if noise_type == "x":
                noise_factors[qubit] = qt.sigmax()
            elif noise_type == "z":
                noise_factors[qubit] = qt.sigmaz()
            else:
                raise ValueError("noise_type must be 'x' or 'z'.")
            collapse_operators.append(np.sqrt(epsilon) * qt.tensor(noise_factors))

    hamiltonian = omega * sum(z_terms)
    return as_scipy_csr(qt.liouvillian(hamiltonian, collapse_operators))


def _validate_time_grid(times, dt, correction_interval):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array.")
    if np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
        raise ValueError("times must be sorted and nonnegative.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    sample_steps = np.rint(times / dt).astype(int)
    represented_times = sample_steps * dt
    tolerance = 1e-10 * max(1.0, float(times[-1]))
    if not np.allclose(times, represented_times, rtol=0.0, atol=tolerance):
        raise ValueError("every requested time must be an integer multiple of dt.")

    correction_steps = None
    if correction_interval is not None:
        correction_interval = float(correction_interval)
        if correction_interval <= 0.0:
            raise ValueError("correction_interval must be positive.")
        correction_steps = int(round(correction_interval / dt))
        if correction_steps < 1 or not np.isclose(
            correction_steps * dt,
            correction_interval,
            rtol=0.0,
            atol=1e-12 * max(1.0, correction_interval),
        ):
            raise ValueError("correction_interval must be an integer multiple of dt.")

    return times, sample_steps, correction_steps


def _simulate_frequency(
    code,
    omega,
    epsilon,
    times,
    sample_steps,
    correction_steps,
    dt,
    noise_type,
    record_second_moment,
):
    n_qubits, psi0, measurement = get_qubit_setup(code)
    liouvillian = _liouvillian(code, n_qubits, omega, epsilon, noise_type)

    measurement_row = as_scipy_csr(qt.operator_to_vector(measurement).dag())
    second_row = None
    if record_second_moment:
        measurement_squared = measurement * measurement
        second_row = as_scipy_csr(
            qt.operator_to_vector(measurement_squared).dag()
        )

    state = as_scipy_csr(qt.operator_to_vector(psi0 * psi0.dag()))
    correction = None
    if correction_steps is not None:
        correction = as_scipy_csr(build_correction_superoperator(code))

    expectations = np.empty(times.size, dtype=float)
    second_moments = (
        np.empty(times.size, dtype=float) if record_second_moment else None
    )

    sample_locations = {}
    for location, step in enumerate(sample_steps):
        sample_locations.setdefault(int(step), []).append(location)

    def record(step):
        for location in sample_locations.get(step, ()):
            expectations[location] = _expectation(measurement_row, state)
            if second_moments is not None:
                second_moments[location] = _expectation(second_row, state)

    record(0)
    for step in range(1, int(sample_steps[-1]) + 1):
        k1 = liouvillian.dot(state)
        k2 = liouvillian.dot(state + dt * k1)
        state = state + 0.5 * dt * (k1 + k2)

        if correction is not None and step % correction_steps == 0:
            state = correction.dot(state)

        record(step)

    return expectations, second_moments


def simulate_projector(
    code: str,
    omega: float,
    epsilon: float,
    times,
    *,
    noise_type: str = "x",
    correction_interval: float | None = None,
    dt: float = 1e-3,
    finite_difference_step: float = 1e-4,
):
    """Simulate projector probability and inverse frequency uncertainty.

    The central frequency ``omega`` is used for the reported probability and
    variance.  The derivative is evaluated independently by the centered
    finite difference at ``omega +/- finite_difference_step``.
    """
    omega = float(omega)
    epsilon = float(epsilon)
    finite_difference_step = float(finite_difference_step)
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative.")
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")

    noise_type = noise_type.lower()
    times, sample_steps, correction_steps = _validate_time_grid(
        times, float(dt), correction_interval
    )

    probability, second_moment = _simulate_frequency(
        code,
        omega,
        epsilon,
        times,
        sample_steps,
        correction_steps,
        float(dt),
        noise_type,
        record_second_moment=True,
    )
    probability_minus, _ = _simulate_frequency(
        code,
        omega - finite_difference_step,
        epsilon,
        times,
        sample_steps,
        correction_steps,
        float(dt),
        noise_type,
        record_second_moment=False,
    )
    probability_plus, _ = _simulate_frequency(
        code,
        omega + finite_difference_step,
        epsilon,
        times,
        sample_steps,
        correction_steps,
        float(dt),
        noise_type,
        record_second_moment=False,
    )

    derivative = (probability_plus - probability_minus) / (
        2.0 * finite_difference_step
    )
    variance = np.clip(second_moment - probability**2, 0.0, None)
    standard_deviation = np.sqrt(variance)
    sensitivity = np.divide(
        np.abs(derivative),
        standard_deviation,
        out=np.zeros_like(derivative),
        where=standard_deviation > 1e-12,
    )

    return {
        "times": times,
        "probability": probability,
        "derivative": derivative,
        "variance": variance,
        "sensitivity": sensitivity,
    }
