# Paper Notebook Guide

These notebooks are organized by workflow: dataset preparation, pressure-model construction, temperature-model construction, figure reproduction, and the Merapi case study.

## Notebook Roles

- `00_independent_dataset.ipynb`: prepares the independent experimental dataset, runs Cpx-liquid equilibrium checks, and generates dataset/calibration comparison figures.
- `01_build_AIMS4PT_cpx_pressure_workflow.ipynb`: builds pressure-model OOD, deviation, and ranking inputs for downstream figures.
- `02_build_AIMS4PT_cpx_temperature_workflow.ipynb`: builds temperature-model OOD, deviation, and ranking inputs for downstream figures.
- `03_reproduce_AIMS4PT_cpx_figures.ipynb`: reproduces SHAP, OOD, TAS-OOD, residual, and regression-slope figures.
- `04_apply_AIMS4PT_cpx_to_Merapi.ipynb`: applies the workflow to Merapi 2006/2010 data and exports case-study figures.

## Figure Map

| Figure | Main notebook | Purpose |
| --- | --- | --- |
| Figure 1 | `00_independent_dataset.ipynb` | Calibration dataset TAS, P-T, and composition summaries. |
| Figure 2 | `03_reproduce_AIMS4PT_cpx_figures.ipynb` | Pressure-model SHAP summary plots. |
| Figure 3 | `03_reproduce_AIMS4PT_cpx_figures.ipynb` | Temperature-model SHAP summary plots. |
| Figure 4 | `03_reproduce_AIMS4PT_cpx_figures.ipynb` | TAS-field and predicted P-T OOD schematic. |
| Figure 5 | `00_independent_dataset.ipynb` | Independent dataset P-T/TAS distributions and calibration comparisons. |
| Figure 6 | `03_reproduce_AIMS4PT_cpx_figures.ipynb` | Feature-space OOD distance versus absolute deviation. |
| Figure 7 | `03_reproduce_AIMS4PT_cpx_figures.ipynb` | Pressure prediction and residual comparisons. |
| Figure 8 | `03_reproduce_AIMS4PT_cpx_figures.ipynb` | Temperature prediction and residual comparisons. |
| Figure 9 | `04_apply_AIMS4PT_cpx_to_Merapi.ipynb` | Merapi Cpx-only and Cpx-liquid P-T estimates. |
| Figure 10 | `04_apply_AIMS4PT_cpx_to_Merapi.ipynb` | Merapi P-T results compared with published constraints. |
