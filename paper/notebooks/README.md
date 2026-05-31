# AIMS4PT_cpx Paper Notebooks

This directory contains notebooks used for manuscript-related analyses, including dataset preparation, AIMS4PT_cpx resource building, figure reproduction, and the Merapi case study.


## Important Warning

Do not run these notebooks unless you understand which intermediate files and trained resources each notebook reads and writes. In particular:

- `02_build_AIMS4PT_cpx_temperature_workflow.ipynb` rebuilds the temperature workflow and may overwrite intermediate or trained resources.
- `03_reproduce_AIMS4PT_cpx_figures.ipynb` depends on outputs from the preceding workflow notebooks and is intended for manuscript figure reproduction, not routine calculation.
- `01_build_AIMS4PT_cpx_pressure_workflow.ipynb` is also a pressure-workflow rebuilding entry point and is not intended for routine use.

## Notebook Roles

- `00_independent_dataset.ipynb`: prepares the independent experimental dataset, runs clinopyroxene-liquid equilibrium checks, and generates dataset/calibration comparison figures.
- `01_build_AIMS4PT_cpx_pressure_workflow.ipynb`: builds pressure-model OOD, deviation, and ranking inputs for downstream figures.
- `02_build_AIMS4PT_cpx_temperature_workflow.ipynb`: builds temperature-model OOD, deviation, and ranking inputs for downstream figures.
- `03_reproduce_AIMS4PT_cpx_figures.ipynb`: reproduces SHAP, OOD, TAS-OOD, residual, and regression-slope figures.
- `04_apply_AIMS4PT_cpx_to_Merapi.ipynb`: applies the workflow to Merapi 2006/2010 data and exports case-study figures.
