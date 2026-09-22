# CSS-codes-for-Quantum-Metrology-with-Discrete-time-Error-Correction

This repository contains the code to generate the plots and the numerical validation of the analytic formulas related to the paper: CSS codes for Quantum Metrology with Discrete-time Error Correction.

## Files

- `generate_paper_figures.py`: main file to generate the plots.
- `analysis.py`: analytical formulas.
- `plotting.py`: plotting function.
- `validation/`: numerical checks.
    - `numerical_simulation.py`: density-matrix evolution with a Liouville master equation using QuTiP and an explicit second-order Runge-Kutta integrator. 
    - `quantum_codes.py`: Steane and Shor states, projector observables, and recovery superoperators used by the numerical simulation.
    - `pauli_algebra.py`: small tensor-product and computational-basis helpers required by quantum_codes.py.
    - `verify_analytics.py`: compares the independent numerical simulation with analysis.py.

## Run

```bash
python generate_paper_figures.py
```

Generated files are written to `plots/`; cached analytical data are written to `sim_data/paper_curves.pkl`.
The generated plot files are:

- `paper_projector_z.pdf`
- `paper_inverse_error_envelope_z.pdf`
- `paper_projector_x.pdf`
- `paper_inverse_error_envelope_x.pdf`
- `paper_HL_SQL_crossover_x.pdf`
- `paper_universal_curve_x.pdf`
- `paper_universal_curve_p_x.pdf`
- `paper_projector_decay_rates.tex`
- `paper_sensitivity_decay_rates.tex`
- `paper_optimal_times.tex`
- `paper_scale_regimes.tex`
