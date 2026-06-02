# AIMS4PT_cpx

This repository contains the code and supporting resources for **AIMS4PT_cpx**, an AI-assisted model-selection framework for clinopyroxene-based pressure-temperature (P-T) estimation. The framework evaluates the applicability of published clinopyroxene-only and clinopyroxene-liquid thermobarometers to a given dataset, runs P-T calculations, and exports reports for model selection and calculation.

## Repository Layout

- `calculator.ipynb`: User-friendly calculator notebook. This is the recommended entry point for applying AIMS4PT_cpx to new datasets.
- `input.xlsx`: Input template for `calculator.ipynb`.
- `src/aims4pt/`: Python source code for AIMS4PT. This includes thermobarometer models, out-of-distribution (OOD) and deviation prediction tools, data-processing utilities, visualization helpers, and Excel report generation.
- `src/aims4pt/model_tools/trained_model/`: Trained AIMS4PT_cpx resources, including OOD detectors, deviation functions, and SHAP results. Standard calculations use these files directly.
- `paper/`: Materials used for manuscript-related analyses, including public datasets and workflow notebooks.


## Installation

Basic use of AIMS4PT_cpx requires at least 4 GB of memory; 8 GB or more is recommended.

Create the conda environment from the provided environment file:

```bash
conda env create -f environment.base.yml
conda activate AIMS4PT
```

Then install the package from the repository root:

```bash
pip install .
```

For development work, manuscript reproduction, or rebuilding AIMS4PT_cpx resources, install in editable mode with the development dependencies:

```bash
pip install -e ".[dev]"
```

## Basic Usage

### Jupyter notebooks

For routine application of the AIMS4PT_cpx framework described in the manuscript, use the root-level `calculator.ipynb` notebook:

1. Install the environment and package following the instructions above.
2. Enter sample data in `input.xlsx`. Clinopyroxene oxide columns should end with `_cpx`, liquid oxide columns should end with `_liq`, and compositions should be reported in wt%.
3. Open and run `calculator.ipynb`.
4. Check the cells marked as `Input required`, including whether clinopyroxene-liquid thermobarometers should be evaluated, the project name, and optional melt TAS information.
5. After execution, results are written to `report_output/<project_name>/`, including Excel reports and intermediate cache files.

Most users do not need to run the notebooks in `paper/notebooks/`.

### Graphical web interface

A lightweight web interface is available for users who prefer a graphical workflow. The web app supports input-file validation, one-click calculation runs, and report downloads from a browser. It can be run locally and is also intended as the basis for future server deployment.

For local use, install the web dependencies:

```bash
pip install ".[web]"
```

Then run the server locally with:

```bash
uvicorn web.main:app --host 127.0.0.1 --port 8003
```

Open a browser and navigate to `http://127.0.0.1:8003 ` to access the interface.

The port can be changed as needed.

## Development and Reproduction

Use editable installation mode if you need to modify the AIMS4PT_cpx framework, recalibrate OOD/deviation/SHAP resources, or reproduce figures:

```bash
pip install -e ".[dev]"
```

**Important warning:** Do not run the notebooks in `paper/notebooks/` unless you understand which intermediate files and trained resources each notebook reads and writes. In particular, `02_build_AIMS4PT_cpx_temperature_workflow.ipynb` rebuilds the temperature workflow, and `03_reproduce_AIMS4PT_cpx_figures.ipynb` depends on outputs from the preceding workflow notebooks to reproduce manuscript figures. Running these notebooks directly may overwrite existing trained resources or intermediate products. `01_build_AIMS4PT_cpx_pressure_workflow.ipynb` is also a pressure-workflow rebuilding entry point and is not intended for routine use.

## Citation

If you use this repository or the AIMS4PT_cpx framework, please cite the manuscript:

> Liu, X.-Y., & Li, W.-R. (in preparation). Ranking Thermobarometers: An Artificial Intelligence-Based Framework for Selecting Clinopyroxene-Based Models.

## Third-Party Code and Packages

This project uses `pyrolite` for Total Alkali-Silica (TAS) classification and TAS-related visualization utilities. Users are encouraged to also cite:

> Williams, M. J., Schoneveld, L., Mao, Y., Klump, J., Gosses, J., Dalton, H., Bath, A., & Barnes, S. (2020). pyrolite: Python for geochemistry. Journal of Open Source Software, 5(50), 2314. https://doi.org/10.21105/joss.02314

This project uses `scikit-learn` for machine learning tools including ExtraTrees, one-class SVM, and nearest neighbors. Users are encouraged to also cite:
> Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12(85), 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html

This project uses `SHAP` for feature-importance analysis. Users are encouraged to also cite:
> Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, 4768–4777.


Small portions of the liquid-mixing routines in this repository were adapted in part from `Thermobar`. If you use or adapt these routines, please also cite:

> Wieser, P., Petrelli, M., Lubbers, J., Wieser, E., Ozaydin, S., Kent, A., & Till, C. (2022). Thermobar: An open-source Python3 tool for thermobarometry and hygrometry. Volcanica, 5(2), 349-384. https://doi.org/10.30909/vol.05.02.349384
