# MQSP manuscript figure code

This directory contains the minimal analytical workflow needed to regenerate the figures and tables referenced by the numerical-analysis TeX section.

## Files

- `generate_paper_figures.py`: configuration, data caching, and the executable figure-generation entry point.
- `analysis.py`: analytical uncorrected dynamics, discrete-QEC projector derivatives, and spectral decay rates used by the figures and tables.
- `plotting.py`: manuscript plotting, envelope fitting, crossover analysis, universal-curve plots, and TeX table generation.

The exploratory full-state simulation files are intentionally excluded because they are not used by the manuscript figure pipeline.

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- A LaTeX installation available to Matplotlib (`text.usetex = True`)

## Run

```bash
python generate_paper_figures.py
```

Generated files are written to `plots/`; cached analytical data are written to `sim_data/paper_curves.pkl`.

The output filenames match the existing TeX references:

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
