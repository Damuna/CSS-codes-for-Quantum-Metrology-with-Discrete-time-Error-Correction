"""Analytical routines used to generate the manuscript figures."""

from functools import lru_cache

import numpy as np
def get_dynamics_and_derivatives(omega, gamma, t, noise_type="x"):
    """Return the local coefficients a, b and their omega derivatives.

    The adjoint evolution is written as
        X(t) = a(t) X - b(t) Y.

    Parameters
    ----------
    noise_type : {"x", "z"}
        Independent local Pauli noise.
    """
    t = np.asarray(t, dtype=float)
    noise = noise_type.lower()

    if noise == "z":
        decay = np.exp(-2.0 * gamma * t)
        phase = 2.0 * omega * t
        a = decay * np.cos(phase)
        b = decay * np.sin(phase)
        da = -2.0 * t * b
        db = 2.0 * t * a
        return a, b, da, db

    if noise != "x":
        raise ValueError("noise_type must be 'x' or 'z'.")

    # X noise.  Use the underdamped expression employed in the manuscript.
    # The critical point is regularized only to avoid a removable numerical
    # singularity in the closed form.
    gamma_eff = float(gamma)
    if np.isclose(gamma_eff, 2.0 * omega, rtol=1e-12, atol=1e-14):
        gamma_eff -= 1e-12

    if gamma_eff < 2.0 * omega:
        Omega = np.sqrt((2.0 * omega) ** 2 - gamma_eff**2)
        dOmega = 4.0 * omega / Omega
        decay = np.exp(-gamma_eff * t)
        C = np.cos(Omega * t)
        S = np.sin(Omega * t)

        a = decay * (C + (gamma_eff / Omega) * S)
        b = decay * ((2.0 * omega / Omega) * S)

        da = (
            decay
            * dOmega
            * (-t * S - (gamma_eff / Omega**2) * S + (gamma_eff * t / Omega) * C)
        )
        db = decay * (
            (-2.0 * gamma_eff**2 / Omega**3) * S
            + (2.0 * omega / Omega) * t * C * dOmega
        )
        return a, b, da, db

    # Overdamped continuation.
    Lambda = np.sqrt(gamma_eff**2 - (2.0 * omega) ** 2)
    dLambda = -4.0 * omega / Lambda
    decay = np.exp(-gamma_eff * t)
    C = np.cosh(Lambda * t)
    S = np.sinh(Lambda * t)

    a = decay * (C + (gamma_eff / Lambda) * S)
    b = decay * ((2.0 * omega / Lambda) * S)
    da = (
        decay
        * dLambda
        * (t * S - (gamma_eff / Lambda**2) * S + (gamma_eff * t / Lambda) * C)
    )
    db = decay * (
        (2.0 * gamma_eff**2 / Lambda**3) * S
        + (2.0 * omega / Lambda) * t * C * dLambda
    )
    return a, b, da, db


def ghz_expectation_and_derivative(omega, gamma, t, n_qubits, noise_type="x"):
    """Parity signal <X^n> and omega derivative for an n-qubit GHZ+ probe."""
    a, b, da, db = get_dynamics_and_derivatives(omega, gamma, t, noise_type=noise_type)
    z = a + 1j * b
    dz = da + 1j * db
    expectation = np.real(z**n_qubits)
    derivative = n_qubits * np.real(z ** (n_qubits - 1) * dz)
    return expectation, derivative


def pauli_sensitivity(expectation, derivative, tol=1e-12):
    """Error-propagation sensitivity for a Pauli observable with outcomes +/-1."""
    expectation = np.asarray(expectation, dtype=float)
    derivative = np.asarray(derivative, dtype=float)
    variance = np.clip(1.0 - expectation**2, 0.0, None)
    denom = np.sqrt(variance)
    out = np.divide(
        np.abs(derivative), denom, out=np.zeros_like(denom), where=denom > tol
    )
    return out


def reference_probe_curves(omega, gamma, t, n_qubits, noise_type="x"):
    """Resource-matched product and GHZ reference probes.

    The product reference uses n independent |+> probes and therefore gains a
    factor sqrt(n) in inverse error.  The GHZ reference uses all n qubits in
    one entangled probe and is read out through X^{\\otimes n}.
    """
    a, _, da, _ = get_dynamics_and_derivatives(omega, gamma, t, noise_type=noise_type)
    single_sens = pauli_sensitivity(a, da)

    ghz_exp, ghz_deriv = ghz_expectation_and_derivative(
        omega, gamma, t, n_qubits, noise_type=noise_type
    )
    ghz_sens = pauli_sensitivity(ghz_exp, ghz_deriv)

    return {
        "product_probability": 0.5 * (1.0 + a),
        "product_sensitivity": np.sqrt(n_qubits) * single_sens,
        "single_sensitivity": single_sens,
        "ghz_probability": 0.5 * (1.0 + ghz_exp),
        "ghz_sensitivity": ghz_sens,
    }


def _parity(x):
    """Return the parity of the integer bit mask x."""
    return int(x.bit_count() & 1)


def _steane_masks():
    """
    Bit masks for the Steane CSS generators used in the manuscript.

    Qubit indices are zero-based versions of
        X1 X3 X5 X7, X2 X3 X6 X7, X4 X5 X6 X7.
    The Z generators have the same supports.
    """
    generators = (
        (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6),
        (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6),
        (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6),
    )

    group = []
    for r in range(8):
        mask = 0
        for i, g in enumerate(generators):
            if (r >> i) & 1:
                mask ^= g
        group.append(mask)

    return generators, tuple(group), tuple(m for m in group if m != 0)


def _steane_recovery_matrix():
    """
    Heisenberg adjoint of Steane bit-flip recovery on the reduced basis
        B_z = X(x) Z(z), z in F_2^7,
    for any fixed nonzero X-stabilizer support x.

    The measured syndrome is that of the three Z-type Steane stabilizers.
    The correction for a nonzero syndrome is the corresponding single-qubit
    X correction determined by the parity-check column.
    """
    z_generators, z_group, _ = _steane_masks()

    # syndrome -> X-correction mask.  Syndrome bits are ordered as the three
    # Z-generator supports above.  The nonzero columns identify qubits 0..6.
    correction = {0: 0}
    for q in range(7):
        syndrome = 0
        for i, g in enumerate(z_generators):
            if (g >> q) & 1:
                syndrome |= 1 << i
        correction[syndrome] = 1 << q

    R = np.zeros((128, 128), dtype=complex)
    for z in range(128):
        for r, h in enumerate(z_group):
            coeff = 0.0
            for s in range(8):
                coeff += (-1) ** (_parity(correction[s] & z) ^ _parity(s & r))
            R[z ^ h, z] += coeff / 8.0
    return R


def _x_noise_coefficients(omega, gamma, duration):
    """
    Single-qubit adjoint coefficients for X noise over one time interval.

    In the basis X, XZ, the Heisenberg map is
        X  -> a X - i b XZ,
        XZ -> -i b X + c XZ,
    while Z -> d Z.  This is equivalent to X -> a X - b Y and
    Y -> b X + c Y.
    """
    s = float(duration)
    if s == 0.0:
        return 1.0, 0.0, 1.0, 1.0

    exp_g = np.exp(-gamma * s)
    if np.isclose(gamma, 2.0 * omega, rtol=1e-12, atol=1e-14):
        a = exp_g * (1.0 + gamma * s)
        b = exp_g * (2.0 * omega * s)
        c = exp_g * (1.0 - gamma * s)
    elif gamma < 2.0 * omega:
        Omega = np.sqrt((2.0 * omega) ** 2 - gamma**2)
        C = np.cos(Omega * s)
        S = np.sin(Omega * s)
        a = exp_g * (C + (gamma / Omega) * S)
        b = exp_g * ((2.0 * omega / Omega) * S)
        c = exp_g * (C - (gamma / Omega) * S)
    else:
        Lambda = np.sqrt(gamma**2 - (2.0 * omega) ** 2)
        C = np.cosh(Lambda * s)
        S = np.sinh(Lambda * s)
        a = exp_g * (C + (gamma / Lambda) * S)
        b = exp_g * ((2.0 * omega / Lambda) * S)
        c = exp_g * (C - (gamma / Lambda) * S)

    d = np.exp(-2.0 * gamma * s)
    return float(a), float(b), float(c), float(d)


def _apply_steane_interval(vec, x_mask, duration, omega, gamma):
    """Apply the exact single-interval adjoint evolution to a coefficient vector."""
    a, b, c, d = _x_noise_coefficients(omega, gamma, duration)
    out = np.zeros(128, dtype=complex)

    support = [q for q in range(7) if (x_mask >> q) & 1]
    not_support = [q for q in range(7) if not ((x_mask >> q) & 1)]

    for z0, lam in enumerate(vec):
        if lam == 0:
            continue

        outside_factor = 1.0
        for q in not_support:
            if (z0 >> q) & 1:
                outside_factor *= d

        # Expand over the 2^4 choices of whether each X/XZ factor toggles.
        for toggle_bits in range(1 << len(support)):
            z1 = z0
            coeff = lam * outside_factor
            for local_idx, q in enumerate(support):
                old_has_z = (z0 >> q) & 1
                toggled = (toggle_bits >> local_idx) & 1
                if toggled:
                    z1 ^= 1 << q
                    coeff *= -1j * b
                else:
                    coeff *= c if old_has_z else a
            out[z1] += coeff

    return out


def _steane_interval_matrix(x_mask, duration, omega, gamma):
    """Dense matrix for one exact single-interval adjoint evolution."""
    E = np.zeros((128, 128), dtype=complex)
    for z in range(128):
        basis = np.zeros(128, dtype=complex)
        basis[z] = 1.0
        E[:, z] = _apply_steane_interval(basis, x_mask, duration, omega, gamma)
    return E


def _apply_interval(vec, x_mask, duration, omega, gamma, n_qubits):
    """Apply exact single-interval adjoint evolution on B_z = X(x) Z(z)."""
    a, b, c, d = _x_noise_coefficients(omega, gamma, duration)
    size = 1 << n_qubits
    out = np.zeros(size, dtype=complex)

    support = [q for q in range(n_qubits) if (x_mask >> q) & 1]
    not_support = [q for q in range(n_qubits) if not ((x_mask >> q) & 1)]

    for z0, lam in enumerate(vec):
        if lam == 0:
            continue

        outside_factor = 1.0
        for q in not_support:
            if (z0 >> q) & 1:
                outside_factor *= d

        for toggle_bits in range(1 << len(support)):
            z1 = z0
            coeff = lam * outside_factor
            for local_idx, q in enumerate(support):
                old_has_z = (z0 >> q) & 1
                toggled = (toggle_bits >> local_idx) & 1
                if toggled:
                    z1 ^= 1 << q
                    coeff *= -1j * b
                else:
                    coeff *= c if old_has_z else a
            out[z1] += coeff

    return out


def _interval_matrix(x_mask, duration, omega, gamma, n_qubits):
    """Dense matrix for one exact single-interval adjoint evolution."""
    size = 1 << n_qubits
    E = np.zeros((size, size), dtype=complex)
    for z in range(size):
        basis = np.zeros(size, dtype=complex)
        basis[z] = 1.0
        E[:, z] = _apply_interval(basis, x_mask, duration, omega, gamma, n_qubits)
    return E


def _shor_recovery_matrix_reduced():
    """
    Reduced Shor recovery matrix for one weight-6 X stabilizer.

    The support consists of two 3-qubit repetition blocks, so only six local
    Z-subset bits are needed.  The unused third block cannot be populated by
    recovery when it is initially absent: summing over its syndrome sector kills
    all Z-stabilizer insertions outside the tracked support.
    """
    z_generators = (
        (1 << 0) | (1 << 1),
        (1 << 1) | (1 << 2),
        (1 << 3) | (1 << 4),
        (1 << 4) | (1 << 5),
    )
    z_group = []
    for r in range(16):
        mask = 0
        for i, g in enumerate(z_generators):
            if (r >> i) & 1:
                mask ^= g
        z_group.append(mask)

    block_correction = {
        0b00: 0,
        0b01: 0,
        0b10: 2,
        0b11: 1,
    }

    correction = {}
    for syndrome in range(16):
        c = 0
        for block in range(2):
            local = (syndrome >> (2 * block)) & 0b11
            if local != 0:
                q = 3 * block + block_correction[local]
                c |= 1 << q
        correction[syndrome] = c

    R = np.zeros((64, 64), dtype=complex)
    for z in range(64):
        for r, h in enumerate(z_group):
            coeff = 0.0
            for s in range(16):
                coeff += (-1) ** (_parity(correction[s] & z) ^ _parity(s & r))
            R[z ^ h, z] += coeff / 16.0
    return R


def _shor_expectation_indicator_reduced():
    """Expectation indicator for two GHZ blocks in the local six-bit basis."""
    indicator = np.zeros(64, dtype=complex)
    block_masks = ((1 << 0) | (1 << 1) | (1 << 2), (1 << 3) | (1 << 4) | (1 << 5))
    for z in range(64):
        if all(_parity(z & block) == 0 for block in block_masks):
            indicator[z] = 1.0
    return indicator


def compute_expectation_values(a, b, code, observable_type):
    """
    Compute expectation values for various observables.

    Parameters:
    -----------
    a, b : np.ndarray
        State coefficients
    code : str
        "Steane" or "Shor"
    observable_type : str
        "single_X", "projector", or "logical_X"

    Returns:
    --------
    exp_val : np.ndarray
        Expectation values
    """
    code_lower = code.lower()

    if observable_type == "single_X":
        # Single qubit <X> expectation
        return a

    elif observable_type == "projector":
        if code_lower == "shor":
            # Projector for Shor code: <Π> = (1 + 3K^2)/4 where K = a^3 - 3ab^2
            K = a**3 - 3 * a * b**2
            return (1 + 3 * K**2) / 4
        elif code_lower == "steane":
            # Projector for Steane code: <Π> = (1 + 7J)/8 where J = a^4 + b^4
            J = a**4 + b**4
            return (1 + 7 * J) / 8
        else:
            raise ValueError(f"Unknown code: {code}")

    elif observable_type == "logical_X":
        if code_lower == "shor":
            # Logical X for Shor code: <X_L> = K^3 where K = a^3 - 3ab^2
            K = a**3 - 3 * a * b**2
            return K**3
        elif code_lower == "steane":
            # Logical X for Steane code: <X_L> = a^7 + 7a^3b^4
            return a**7 + 7 * (a**3) * (b**4)
        else:
            raise ValueError(f"Unknown code: {code}")

    else:
        raise ValueError(
            f"Unknown observable_type: {observable_type}. "
            "Must be one of: 'single_X', 'projector', 'logical_X'"
        )


def compute_sensitivity(a, b, da, db, code, observable_type):
    """
    Compute sensitivity 1/δω for various observables.

    Parameters:
    -----------
    a, b, da, db : np.ndarray
        State coefficients and their derivatives
    code : str
        "Steane" or "Shor"
    observable_type : str
        "single_X", "projector", or "logical_X"

    Returns:
    --------
    sensitivity : np.ndarray
        Sensitivity values 1/δω
    """
    code_lower = code.lower()

    # Compute expectation value and its derivative
    if observable_type == "single_X":
        exp_val = a
        d_exp_val = da
        # Standard deviation for single qubit Pauli: sqrt(1 - <X>^2)
        delta = np.sqrt(1 - exp_val**2)

    elif observable_type == "projector":
        if code_lower == "shor":
            K = a**3 - 3 * a * b**2
            dK = 3 * (a**2 - b**2) * da - 6 * a * b * db
            exp_val = (1 + 3 * K**2) / 4
            d_exp_val = (3 / 2) * K * dK
            # Standard deviation for projector: sqrt(<Π>(1-<Π>))
            delta = np.sqrt(exp_val * (1 - exp_val))

        elif code_lower == "steane":
            J = a**4 + b**4
            dJ = 4 * a**3 * da + 4 * b**3 * db
            exp_val = (1 + 7 * J) / 8
            d_exp_val = (7 / 8) * dJ
            # Standard deviation for projector: sqrt(<Π>(1-<Π>))
            delta = np.sqrt(exp_val * (1 - exp_val))

        else:
            raise ValueError(f"Unknown code: {code}")

    elif observable_type == "logical_X":
        if code_lower == "shor":
            K = a**3 - 3 * a * b**2
            dK = 3 * (a**2 - b**2) * da - 6 * a * b * db
            exp_val = K**3
            d_exp_val = 3 * K**2 * dK

        elif code_lower == "steane":
            exp_val = a**7 + 7 * (a**3) * (b**4)
            # Derivative using product rule
            d_term1 = 7 * a**6 * da
            d_term2 = 7 * (3 * a**2 * da * b**4 + a**3 * 4 * b**3 * db)
            d_exp_val = d_term1 + d_term2

        else:
            raise ValueError(f"Unknown code: {code}")

        # Standard deviation for logical Pauli: sqrt(1 - <X_L>^2)
        delta = np.sqrt(1 - exp_val**2)

    else:
        raise ValueError(
            f"Unknown observable_type: {observable_type}. "
            "Must be one of: 'single_X', 'projector', 'logical_X'"
        )

    # Compute sensitivity: |d<O>/dω| / ΔO
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.abs(d_exp_val) / delta
        sensitivity[delta < 1e-10] = 0.0

    return sensitivity


def compute_single_qubit_sensitivity(a, da, n_qubits=None):
    """
    Compute SQL sensitivity for single qubit measurement.

    Parameters:
    -----------
    a, da : np.ndarray
        Single qubit state coefficient and derivative
    n_qubits : int, optional
        Number of qubits for SQL scaling (sqrt(N))

    Returns:
    --------
    sensitivity : np.ndarray
        Sensitivity values 1/δω
    """
    # Standard deviation for single qubit Pauli: sqrt(1 - <X>^2)
    delta_X = np.sqrt(1 - a**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.abs(da) / delta_X
        sens[delta_X < 1e-10] = 0.0

    # Apply SQL scaling if requested
    if n_qubits is not None:
        sens *= np.sqrt(n_qubits)

    return sens


def compute_all_analytics(omega, gamma, t, code, noise_type="x"):
    """Compute uncorrected code and reference-probe analytics."""
    a, b, da, db = get_dynamics_and_derivatives(omega, gamma, t, noise_type=noise_type)
    n_qubits = 9 if code.lower() == "shor" else 7

    results = {
        "t": np.asarray(t, dtype=float),
        "a": a,
        "b": b,
        "da": da,
        "db": db,
        "n_qubits": n_qubits,
        "noise_type": noise_type.lower(),
    }

    observable_types = ["single_X", "projector", "logical_X"]
    for obs_type in observable_types:
        results[f"exp_{obs_type}"] = compute_expectation_values(a, b, code, obs_type)

    for obs_type in observable_types:
        if obs_type == "single_X":
            results[f"sens_{obs_type}"] = compute_single_qubit_sensitivity(
                a, da, n_qubits
            )
        else:
            results[f"sens_{obs_type}"] = compute_sensitivity(
                a, b, da, db, code, obs_type
            )

    results["sens_single_unscaled"] = compute_single_qubit_sensitivity(a, da)
    results.update(
        reference_probe_curves(omega, gamma, t, n_qubits, noise_type=noise_type)
    )
    return results


def _x_noise_coefficients_and_derivatives(omega, gamma, duration):
    """
    Single-qubit X-noise coefficients and analytic omega-derivatives.

    Matches the convention used by _apply_steane_interval:
        X  -> a X - i b XZ
        XZ -> -i b X + c XZ
    """
    s = float(duration)

    if s == 0.0:
        return 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0

    if np.isclose(gamma, 2.0 * omega, rtol=1e-12, atol=1e-14):
        gamma = gamma - 1e-12

    exp_g = np.exp(-gamma * s)

    if gamma < 2.0 * omega:
        Omega = np.sqrt((2.0 * omega) ** 2 - gamma**2)
        dOmega = 4.0 * omega / Omega

        C = np.cos(Omega * s)
        S = np.sin(Omega * s)

        a = exp_g * (C + (gamma / Omega) * S)
        b = exp_g * ((2.0 * omega / Omega) * S)
        c = exp_g * (C - (gamma / Omega) * S)

        da = (
            exp_g
            * dOmega
            * (-s * S - (gamma / Omega**2) * S + (gamma * s / Omega) * C)
        )

        db = exp_g * (
            (-2.0 * gamma**2 / Omega**3) * S
            + (2.0 * omega / Omega) * s * C * dOmega
        )

        dc = (
            exp_g
            * dOmega
            * (-s * S + (gamma / Omega**2) * S - (gamma * s / Omega) * C)
        )

    else:
        Lambda = np.sqrt(gamma**2 - (2.0 * omega) ** 2)
        dLambda = -4.0 * omega / Lambda

        C = np.cosh(Lambda * s)
        S = np.sinh(Lambda * s)

        a = exp_g * (C + (gamma / Lambda) * S)
        b = exp_g * ((2.0 * omega / Lambda) * S)
        c = exp_g * (C - (gamma / Lambda) * S)

        da = (
            exp_g
            * dLambda
            * (s * S - (gamma / Lambda**2) * S + (gamma * s / Lambda) * C)
        )

        db = exp_g * (
            (2.0 * gamma**2 / Lambda**3) * S
            + (2.0 * omega / Lambda) * s * C * dLambda
        )

        dc = (
            exp_g
            * dLambda
            * (s * S + (gamma / Lambda**2) * S - (gamma * s / Lambda) * C)
        )

    d = np.exp(-2.0 * gamma * s)
    dd = 0.0

    return tuple(map(float, (a, b, c, d, da, db, dc, dd)))


def _apply_steane_interval_and_derivative(vec, x_mask, duration, omega, gamma):
    """
    Apply one exact Steane interval and its analytic omega-derivative.

    Returns:
        out  = E vec
        dout = (partial_omega E) vec
    """
    a, b, c, d, da, db, dc, _ = _x_noise_coefficients_and_derivatives(
        omega, gamma, duration
    )

    out = np.zeros(128, dtype=complex)
    dout = np.zeros(128, dtype=complex)

    support = [q for q in range(7) if (x_mask >> q) & 1]
    not_support = [q for q in range(7) if not ((x_mask >> q) & 1)]

    for z0, lam in enumerate(vec):
        if lam == 0:
            continue

        outside_factor = 1.0
        for q in not_support:
            if (z0 >> q) & 1:
                outside_factor *= d

        for toggle_bits in range(1 << len(support)):
            z1 = z0
            coeff = lam * outside_factor
            dcoeff = 0.0j

            for local_idx, q in enumerate(support):
                old_has_z = (z0 >> q) & 1
                toggled = (toggle_bits >> local_idx) & 1

                if toggled:
                    z1 ^= 1 << q
                    factor = -1j * b
                    dfactor = -1j * db
                else:
                    if old_has_z:
                        factor = c
                        dfactor = dc
                    else:
                        factor = a
                        dfactor = da

                dcoeff = dcoeff * factor + coeff * dfactor
                coeff *= factor

            out[z1] += coeff
            dout[z1] += dcoeff

    return out, dout


def _steane_interval_matrix_and_derivative(x_mask, duration, omega, gamma):
    """Dense matrix E and analytic derivative partial_omega E."""
    E = np.zeros((128, 128), dtype=complex)
    dE = np.zeros((128, 128), dtype=complex)

    for z in range(128):
        basis = np.zeros(128, dtype=complex)
        basis[z] = 1.0
        E[:, z], dE[:, z] = _apply_steane_interval_and_derivative(
            basis, x_mask, duration, omega, gamma
        )

    return E, dE


def _power_derivative_contraction(
    eigvals, cycles, left, dA_eig, coeffs, chunk_size=64, tol=1e-12
):
    """Contract partial_omega A^n with left/right vectors in chunks.

    This evaluates
        sum_ij left_i S_ij(n) (dA)_ij coeff_j
    for many different cycle counts without forming one large 3-D array for the
    full time grid.
    """
    eigvals = np.asarray(eigvals, dtype=complex)
    cycles = np.asarray(cycles, dtype=np.int64)
    left = np.asarray(left, dtype=complex)
    coeffs = np.asarray(coeffs, dtype=complex)

    lam_i = eigvals[:, None]
    lam_j = eigvals[None, :]
    diff = lam_i - lam_j
    repeated = np.abs(diff) < tol
    kernel = left[:, None] * dA_eig

    out = np.empty(cycles.size, dtype=complex)
    for start in range(0, cycles.size, chunk_size):
        stop = min(start + chunk_size, cycles.size)
        n = cycles[start:stop]
        S = np.empty((len(n), eigvals.size, eigvals.size), dtype=complex)

        # Generic divided difference of z^n.
        pi = eigvals[None, :, None] ** n[:, None, None]
        pj = eigvals[None, None, :] ** n[:, None, None]
        np.divide(pi - pj, diff[None, :, :], out=S, where=~repeated[None, :, :])

        if np.any(repeated):
            ii, jj = np.where(repeated)
            for i, j in zip(ii, jj):
                lam = eigvals[i]
                vals = np.zeros(len(n), dtype=complex)
                positive = n > 0
                vals[positive] = n[positive] * lam ** (n[positive] - 1)
                S[:, i, j] = vals

        out[start:stop] = np.einsum(
            "ij,tij,jt->t",
            kernel,
            S,
            coeffs[:, start:stop],
            optimize=True,
        )
    return out


def discrete_x_spectral_decay_rates(
    code,
    omega,
    gamma,
    tau,
    overlap_tol=1e-10,
    modulus_rtol=1e-10,
    modulus_atol=1e-14,
):
    """
    Extract the distinct exponential decay rates of the measured projector
    directly from the eigenvalues of one discrete-QEC update cycle.

    At exact correction times t = n*tau, the relevant projector contribution is

        mu_n = q^T A^n v0
             = sum_j c_j lambda_j^n,

    where A is the adjoint one-cycle update matrix, lambda_j are its
    eigenvalues, and

        c_j = (q^T v_j) (V^{-1} v0)_j.

    Only modes with non-negligible |c_j| contribute to the measured projector.
    Writing lambda_j = |lambda_j| exp(i phi_j),

        lambda_j^n
            = exp[-D_j t] exp[i Omega_j t],

        D_j     = -log(|lambda_j|)/tau,
        Omega_j = arg(lambda_j)/tau.

    For the Steane X-projector with discrete X-error correction there is one
    relevant real eigenmode and one relevant complex-conjugate pair.  The pair
    has one common modulus, so there are two distinct decay rates.

    Parameters
    ----------
    code : str
        "Steane" or "Shor".
    omega, gamma, tau : float
        Signal frequency, X-noise rate, and correction period.
    overlap_tol : float
        Relative threshold for discarding eigenmodes with negligible projector
        overlap.
    modulus_rtol, modulus_atol : float
        Tolerances used to group modes with the same eigenvalue modulus
        (e.g. a complex-conjugate pair).

    Returns
    -------
    result : dict
        Contains D1 and D2 (ordered from slow to fast), plus the grouped
        spectral modes used to obtain them.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if omega < 0 or gamma < 0:
        raise ValueError("omega and gamma must be nonnegative.")

    code_lower = code.lower()

    if code_lower == "steane":
        _, z_group, x_group_nonzero = _steane_masks()

        q = np.zeros(128, dtype=complex)
        q[list(z_group)] = 1.0

        x_mask = x_group_nonzero[0]
        R = _steane_recovery_matrix()
        E_tau = _steane_interval_matrix(x_mask, tau, omega, gamma)
        A = E_tau @ R

    elif code_lower == "shor":
        q = _shor_expectation_indicator_reduced()

        x_mask = (1 << 6) - 1
        R = _shor_recovery_matrix_reduced()
        E_tau = _interval_matrix(x_mask, tau, omega, gamma, 6)
        A = E_tau @ R

    else:
        raise ValueError(
            "Spectral decay rates are implemented for Steane and Shor codes."
        )

    eigvals, eigvecs = np.linalg.eig(A)

    v0 = np.zeros(A.shape[0], dtype=complex)
    v0[0] = 1.0

    left = q @ eigvecs
    try:
        right = np.linalg.solve(eigvecs, v0)
    except np.linalg.LinAlgError:
        right = np.linalg.pinv(eigvecs) @ v0

    overlaps = left * right
    overlap_scale = float(np.max(np.abs(overlaps)))
    if overlap_scale == 0.0:
        raise RuntimeError(
            "No update-matrix eigenmode overlaps the measured projector."
        )

    relevant = np.abs(overlaps) > overlap_tol * overlap_scale

    modes = []
    for lam, coeff in zip(eigvals[relevant], overlaps[relevant]):
        modulus = float(np.abs(lam))
        if modulus <= 0.0:
            continue

        # Suppress an unphysical negative decay rate caused only by roundoff
        # when a CPTP/adjoint eigenvalue lands infinitesimally outside the unit disk.
        if modulus > 1.0 and np.isclose(modulus, 1.0, rtol=0.0, atol=1e-12):
            modulus = 1.0

        modes.append(
            {
                "eigenvalue": complex(lam),
                "modulus": modulus,
                "decay_rate": float(-np.log(modulus) / tau),
                "frequency": float(np.angle(lam) / tau),
                "overlap": complex(coeff),
            }
        )

    if not modes:
        raise RuntimeError("No nonzero relevant update-matrix eigenvalues were found.")

    # Group modes by modulus.  A complex-conjugate pair therefore gives one D.
    groups = []
    for mode in sorted(modes, key=lambda m: m["modulus"], reverse=True):
        for group in groups:
            if np.isclose(
                mode["modulus"],
                group["modulus"],
                rtol=modulus_rtol,
                atol=modulus_atol,
            ):
                group["modes"].append(mode)
                weights = np.array(
                    [abs(m["overlap"]) for m in group["modes"]], dtype=float
                )
                moduli = np.array([m["modulus"] for m in group["modes"]], dtype=float)
                group["modulus"] = float(np.average(moduli, weights=weights))
                group["decay_rate"] = float(-np.log(group["modulus"]) / tau)
                group["total_overlap"] = float(np.sum(weights))
                break
        else:
            groups.append(
                {
                    "modulus": mode["modulus"],
                    "decay_rate": mode["decay_rate"],
                    "total_overlap": float(abs(mode["overlap"])),
                    "modes": [mode],
                }
            )

    # Ignore any tiny numerical groups that survived the individual-mode cutoff.
    group_scale = max(g["total_overlap"] for g in groups)
    groups = [g for g in groups if g["total_overlap"] > overlap_tol * group_scale]
    groups.sort(key=lambda g: g["decay_rate"])

    result = {
        "D1": float(groups[0]["decay_rate"]),
        "lambda1_modulus": float(groups[0]["modulus"]),
        "groups": groups,
    }
    if len(groups) > 1:
        result["D2"] = float(groups[1]["decay_rate"])
        result["lambda2_modulus"] = float(groups[1]["modulus"])
    else:
        result["D2"] = None
        result["lambda2_modulus"] = None
    return result


def discrete_x_projector_decay_rate_series(code, omega, gamma, tau, order=3):
    """Small-tau expansion of the stroboscopic projector decay rates.

    The returned rates approximate the exact spectral quantities

        D_j(tau) = -log|lambda_j(tau)| / tau

    obtained from :func:`discrete_x_spectral_decay_rates`.  ``order`` denotes
    the highest power of ``tau`` retained in the decay-rate expansion itself.

    For Steane, ``real`` is the nonoscillatory mode and ``osc`` the complex
    conjugate pair.  For Shor there is one common modulus, returned as
    ``common``.
    """
    tau = float(tau)
    omega = float(omega)
    gamma = float(gamma)
    order = int(order)

    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3.")
    if tau < 0 or gamma < 0:
        raise ValueError("tau and gamma must be nonnegative.")

    if code.lower() == "steane":
        D_real = 6.0 * gamma**2 * tau
        D_osc = 18.0 * gamma**2 * tau

        if order >= 2:
            D_real += ((32.0 / 3.0) * gamma * omega**2 - 32.0 * gamma**3) * tau**2
            D_osc += ((32.0 / 3.0) * gamma * omega**2 - 92.0 * gamma**3) * tau**2

        if order >= 3:
            if np.isclose(omega, 0.0, rtol=0.0, atol=1e-15):
                raise ValueError(
                    "The O(tau^3) Steane branch expansion assumes omega != 0."
                )
            D_real += (
                176.0 * gamma**4
                - (136.0 / 3.0) * gamma**2 * omega**2
                - (81.0 / 4.0) * gamma**6 / omega**2
            ) * tau**3
            D_osc += (
                474.0 * gamma**4
                - 104.0 * gamma**2 * omega**2
                + (81.0 / 8.0) * gamma**6 / omega**2
            ) * tau**3

        return {"real": float(D_real), "osc": float(D_osc)}

    if code.lower() == "shor":
        D = 6.0 * gamma**2 * tau
        if order >= 2:
            D += (16.0 * gamma * omega**2 - 16.0 * gamma**3) * tau**2
        if order >= 3:
            D += (44.0 * gamma**4 - 80.0 * gamma**2 * omega**2) * tau**3
        return {"common": float(D)}

    raise ValueError("code must be 'Steane' or 'Shor'.")


def steane_projector_discrete_with_derivative(t, omega, gamma, tau):
    """Corrected Steane projector and analytic omega derivative."""
    scalar_input = np.isscalar(t)
    original_shape = np.shape(t)
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()

    if gamma == 0:
        projector = 25.0 / 32.0 + (7.0 / 32.0) * np.cos(8.0 * omega * t)
        d_projector = -(7.0 / 4.0) * t * np.sin(8.0 * omega * t)
        if scalar_input:
            return float(projector[0]), float(d_projector[0])
        return projector.reshape(original_shape), d_projector.reshape(original_shape)

    if tau <= 0:
        raise ValueError("tau must be positive.")

    _, z_group, x_group_nonzero = _steane_masks()
    z_indicator = np.zeros(128, dtype=complex)
    z_indicator[list(z_group)] = 1.0
    R = _steane_recovery_matrix()

    cycles = np.floor((t + 1e-12 * tau) / tau).astype(np.int64)
    remainders = t - cycles * tau
    remainders[np.abs(remainders) < 1e-13 * tau] = 0.0

    x_mask = x_group_nonzero[0]
    E_tau, dE_tau = _steane_interval_matrix_and_derivative(x_mask, tau, omega, gamma)
    A = E_tau @ R
    dA = dE_tau @ R

    eigvals, eigvecs = np.linalg.eig(A)
    try:
        eigvecs_inv = np.linalg.inv(eigvecs)
    except np.linalg.LinAlgError:
        eigvecs_inv = np.linalg.pinv(eigvecs)

    left = z_indicator @ eigvecs
    dA_eig = eigvecs_inv @ dA @ eigvecs

    e0 = np.zeros(128, dtype=complex)
    e0[0] = 1.0
    V_rem = np.empty((128, t.size), dtype=complex)
    dV_rem = np.empty((128, t.size), dtype=complex)
    for idx, r in enumerate(remainders):
        V_rem[:, idx], dV_rem[:, idx] = _apply_steane_interval_and_derivative(
            e0, x_mask, float(r), omega, gamma
        )

    coeffs = eigvecs_inv @ V_rem
    dcoeffs = eigvecs_inv @ dV_rem
    powers = eigvals[:, None] ** cycles[None, :]

    mu = np.sum(left[:, None] * powers * coeffs, axis=0)
    dmu_remainder = np.sum(left[:, None] * powers * dcoeffs, axis=0)
    dmu_power = _power_derivative_contraction(eigvals, cycles, left, dA_eig, coeffs)

    projector = np.real_if_close((1.0 + 7.0 * mu) / 8.0, tol=1000).real
    d_projector = np.real_if_close(
        7.0 * (dmu_remainder + dmu_power) / 8.0, tol=1000
    ).real
    projector = np.clip(projector, 0.0, 1.0)

    if scalar_input:
        return float(projector[0]), float(d_projector[0])
    return projector.reshape(original_shape), d_projector.reshape(original_shape)


def _apply_interval_and_derivative_generic(
    vec, x_mask, duration, omega, gamma, n_qubits
):
    """Apply E and partial_omega E on B_z = X(x)Z(z)."""
    a, b, c, d, da, db, dc, _ = _x_noise_coefficients_and_derivatives(
        omega, gamma, duration
    )
    size = 1 << n_qubits
    out = np.zeros(size, dtype=complex)
    dout = np.zeros(size, dtype=complex)

    support = [q for q in range(n_qubits) if (x_mask >> q) & 1]
    outside = [q for q in range(n_qubits) if not ((x_mask >> q) & 1)]

    for z0, lam in enumerate(vec):
        if lam == 0:
            continue

        outside_factor = 1.0
        for q in outside:
            if (z0 >> q) & 1:
                outside_factor *= d

        for toggle_bits in range(1 << len(support)):
            z1 = z0
            coeff = lam * outside_factor
            dcoeff = 0.0j

            for local_idx, q in enumerate(support):
                old_has_z = (z0 >> q) & 1
                toggled = (toggle_bits >> local_idx) & 1

                if toggled:
                    z1 ^= 1 << q
                    factor = -1j * b
                    dfactor = -1j * db
                elif old_has_z:
                    factor = c
                    dfactor = dc
                else:
                    factor = a
                    dfactor = da

                dcoeff = dcoeff * factor + coeff * dfactor
                coeff *= factor

            out[z1] += coeff
            dout[z1] += dcoeff

    return out, dout


def _interval_matrix_and_derivative_generic(x_mask, duration, omega, gamma, n_qubits):
    size = 1 << n_qubits
    E = np.zeros((size, size), dtype=complex)
    dE = np.zeros((size, size), dtype=complex)
    for z in range(size):
        basis = np.zeros(size, dtype=complex)
        basis[z] = 1.0
        E[:, z], dE[:, z] = _apply_interval_and_derivative_generic(
            basis, x_mask, duration, omega, gamma, n_qubits
        )
    return E, dE


def shor_projector_discrete_with_derivative(t, omega, gamma, tau):
    """Exact Shor X-projector and omega derivative under discrete X-QEC."""
    scalar_input = np.isscalar(t)
    original_shape = np.shape(t)
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()

    if np.any(t < 0):
        raise ValueError("t must be nonnegative.")
    if tau <= 0:
        raise ValueError("tau must be positive.")

    if gamma == 0:
        phase = 6.0 * omega * t
        projector = (1.0 + 3.0 * np.cos(phase) ** 2) / 4.0
        d_projector = -(9.0 / 2.0) * t * np.sin(2.0 * phase)
        if scalar_input:
            return float(projector[0]), float(d_projector[0])
        return projector.reshape(original_shape), d_projector.reshape(original_shape)

    expectation = _shor_expectation_indicator_reduced()
    R = _shor_recovery_matrix_reduced()
    x_mask = (1 << 6) - 1

    E_tau, dE_tau = _interval_matrix_and_derivative_generic(
        x_mask, tau, omega, gamma, 6
    )
    A = E_tau @ R
    dA = dE_tau @ R

    eigvals, eigvecs = np.linalg.eig(A)
    try:
        eigvecs_inv = np.linalg.inv(eigvecs)
    except np.linalg.LinAlgError:
        eigvecs_inv = np.linalg.pinv(eigvecs)

    left = expectation @ eigvecs
    dA_eig = eigvecs_inv @ dA @ eigvecs
    cycles = np.floor((t + 1e-12 * tau) / tau).astype(np.int64)
    remainders = t - cycles * tau
    remainders[np.abs(remainders) < 1e-13 * tau] = 0.0

    e0 = np.zeros(64, dtype=complex)
    e0[0] = 1.0
    V_rem = np.empty((64, t.size), dtype=complex)
    dV_rem = np.empty((64, t.size), dtype=complex)
    for idx, r in enumerate(remainders):
        V_rem[:, idx], dV_rem[:, idx] = _apply_interval_and_derivative_generic(
            e0, x_mask, float(r), omega, gamma, 6
        )

    coeffs = eigvecs_inv @ V_rem
    dcoeffs = eigvecs_inv @ dV_rem
    powers = eigvals[:, None] ** cycles[None, :]

    mu = np.sum(left[:, None] * powers * coeffs, axis=0)
    dmu_remainder = np.sum(left[:, None] * powers * dcoeffs, axis=0)
    dmu_power = _power_derivative_contraction(eigvals, cycles, left, dA_eig, coeffs)

    projector = np.real_if_close((1.0 + 3.0 * mu) / 4.0, tol=1000).real
    d_projector = np.real_if_close(
        3.0 * (dmu_remainder + dmu_power) / 4.0, tol=1000
    ).real
    projector = np.clip(projector, 0.0, 1.0)

    if scalar_input:
        return float(projector[0]), float(d_projector[0])
    return projector.reshape(original_shape), d_projector.reshape(original_shape)


def projector_discrete_with_derivative(code, t, omega, gamma, tau):
    """Exact discrete-X-QEC projector and derivative for Steane or Shor."""
    code_lower = code.lower()
    if code_lower == "steane":
        return steane_projector_discrete_with_derivative(t, omega, gamma, tau)
    if code_lower == "shor":
        return shor_projector_discrete_with_derivative(t, omega, gamma, tau)
    raise ValueError(f"Unsupported code: {code}")


def corrected_projector_and_sensitivity(code, t, omega, gamma, tau):
    """Return projector probability and inverse error for discrete X-QEC."""
    projector, derivative = projector_discrete_with_derivative(
        code, t, omega, gamma, tau
    )
    variance = np.clip(projector * (1.0 - projector), 0.0, None)
    denom = np.sqrt(variance)
    sensitivity = np.divide(
        np.abs(derivative),
        denom,
        out=np.zeros_like(np.asarray(projector, dtype=float)),
        where=denom > 1e-12,
    )
    return projector, sensitivity
