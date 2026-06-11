# AIMS4PT_cpx

This repository contains the code and supporting resources for **AIMS4PT_cpx**, an AI-assisted model-selection framework for clinopyroxene-based pressure–temperature (P–T) estimation. The framework evaluates the applicability of published clinopyroxene-only and clinopyroxene–liquid thermobarometers to a given dataset, runs P–T calculations, and exports reports for model selection and calculation.

## Repository Layout

- `calculator.ipynb`: User-friendly calculator notebook. This is the recommended entry point for applying AIMS4PT_cpx to new datasets.
- `input.xlsx`: Input template for `calculator.ipynb`.
- `src/aims4pt/`: Python source code for AIMS4PT. This includes thermobarometer models, out-of-distribution (OOD) and deviation prediction tools, data-processing utilities, visualization helpers, and Excel report generation.
- `src/aims4pt/model_tools/trained_model/`: Trained AIMS4PT_cpx resources, including OOD detectors, deviation predictors, and SHAP results. Standard calculations use these files directly.
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

## Basic Usage

### Jupyter notebooks

For routine application of the AIMS4PT_cpx framework described in the manuscript, use the root-level `calculator.ipynb` notebook:

1. Install the environment and package following the instructions above.
2. Enter sample data in `input.xlsx`. Clinopyroxene oxide columns should end with `_cpx`, liquid oxide columns should end with `_liq`, and compositions should be reported in wt%.
3. Open and run `calculator.ipynb`.
4. Check the cells marked as `Input required`, including whether clinopyroxene–liquid thermobarometers should be evaluated, the project name, and optional melt TAS information.
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
python -m aims4pt_web.launcher
```

The launcher opens the interface in your default browser after the server is ready.

## Development and Reproduction

Use editable installation mode if you need to modify the AIMS4PT_cpx framework, recalibrate OOD/deviation/SHAP resources, or reproduce figures:

```bash
pip install -e ".[dev]"
```

**Important warning:** Do not run the notebooks in `paper/notebooks/` unless you understand which intermediate files and trained resources each notebook reads and writes. In particular, `02_build_AIMS4PT_cpx_temperature_workflow.ipynb` rebuilds the temperature workflow, and `03_reproduce_AIMS4PT_cpx_figures.ipynb` depends on outputs from the preceding workflow notebooks to reproduce manuscript figures. Running these notebooks directly may overwrite existing trained resources or intermediate products. `01_build_AIMS4PT_cpx_pressure_workflow.ipynb` is also a pressure-workflow rebuilding entry point and is not intended for routine use.

## Citation

If you use this repository or the AIMS4PT_cpx framework, please cite the manuscript:

> Liu, X.-Y., & Li, W.-R. (in preparation). Ranking Thermobarometers: An Artificial Intelligence-Based Framework for Selecting Clinopyroxene-Based Models.

AIMS4PT_cpx evaluates and applies published clinopyroxene-based thermobarometers. If you use P–T estimates from a model selected or evaluated by AIMS4PT_cpx, please also cite the original source(s) of the corresponding thermobarometer model(s).

### Thermobarometer Model References

| Model                      | Model type                               | AIMS4PT class      | Module                                 |
| -------------------------- | ---------------------------------------- | ------------------ | -------------------------------------- |
| Putirka (2008)             | Clinopyroxene-only; clinopyroxene–liquid | `Putirka_08`       | `aims4pt.model_tools.Putirka_08`       |
| Neave & Putirka (2017)     | Clinopyroxene–liquid                     | `Neave_Putirka_17` | `aims4pt.model_tools.Neave_Putirka_17` |
| Brugman & Till (2019)      | Clinopyroxene–liquid                     | `Brugman_Till_19`  | `aims4pt.model_tools.Brugman_Till_19`  |
| Petrelli et al. (2020)     | Clinopyroxene-only; clinopyroxene–liquid | `Petrelli20`       | `aims4pt.model_tools.Petrelli20`       |
| Wang et al. (2021)         | Clinopyroxene-only                       | `Wang21`           | `aims4pt.model_tools.Wang_21`          |
| Higgins et al. (2021)      | Clinopyroxene-only                       | `Higgins21`        | `aims4pt.model_tools.Higgins21`        |
| Jorgenson et al. (2022)    | Clinopyroxene-only; clinopyroxene–liquid | `Jorgenson22`      | `aims4pt.model_tools.Jorgenson22`      |
| Chicchi et al. (2023)      | Clinopyroxene-only; clinopyroxene–liquid | `Chicchi23`        | `aims4pt.model_tools.Chicchi23`        |
| Ágreda-López et al. (2024) | Clinopyroxene-only; clinopyroxene–liquid | `Agreda2024`       | `aims4pt.model_tools.Agreda2024`       |


Full references for the thermobarometer models are provided below for convenience:

> Ágreda-López, M., Parodi, V., Musu, A., Jorgenson, C., Carfì, A., Mastrogiovanni, F., Caricchi, L., Perugini, D., & Petrelli, M. (2024). Enhancing machine learning thermobarometry for clinopyroxene-bearing magmas. *Computers & Geosciences*, 193, 105707. https://doi.org/10.1016/j.cageo.2024.105707

> Brugman, K. K., & Till, C. B. (2019). A low-aluminum clinopyroxene-liquid geothermometer for high-silica magmatic systems. *American Mineralogist*, 104(7), 996–1004. https://doi.org/10.2138/am-2019-6842

> Chicchi, L., Bindi, L., Fanelli, D., & Tommasini, S. (2023). Frontiers of thermobarometry: GAIA, a novel Deep Learning-based tool for volcano plumbing systems. *Earth and Planetary Science Letters*, 620, 118352. https://doi.org/10.1016/j.epsl.2023.118352

> Higgins, O., Sheldrake, T., & Caricchi, L. (2021). Machine learning thermobarometry and chemometry using amphibole and clinopyroxene: A window into the roots of an arc volcano (Mount Liamuiga, Saint Kitts). *Contributions to Mineralogy and Petrology*, 177, 1. https://doi.org/10.1007/s00410-021-01874-6

> Jorgenson, C., Higgins, O., Petrelli, M., Bégué, F., & Caricchi, L. (2022). A machine learning-based approach to clinopyroxene thermobarometry: Model optimization and distribution for use in Earth sciences. *Journal of Geophysical Research: Solid Earth*, 127(4), e2021JB022904. https://doi.org/10.1029/2021JB022904

> Neave, D. A., & Putirka, K. D. (2017). A new clinopyroxene-liquid barometer, and implications for magma storage pressures under Icelandic rift zones. *American Mineralogist*, 102(4), 777–794. https://doi.org/10.2138/am-2017-5968

> Petrelli, M., Caricchi, L., & Perugini, D. (2020). Machine learning thermo-barometry: Application to clinopyroxene-bearing magmas. *Journal of Geophysical Research: Solid Earth*, 125(9), e2020JB020130. https://doi.org/10.1029/2020JB020130

> Putirka, K. D. (2008). Thermometers and barometers for volcanic systems. *Reviews in Mineralogy and Geochemistry*, 69(1), 61–120. https://doi.org/10.2138/rmg.2008.69.3

> Wang, X., Hou, T., Wang, M., Zhang, C., Zhang, Z., Pan, R., Marxer, F., & Zhang, H. (2021). A new clinopyroxene thermobarometer for mafic to intermediate magmatic systems. *European Journal of Mineralogy*, 33(5), 621–637. https://doi.org/10.5194/ejm-33-621-2021

### Third-Party Code and Packages

This project uses `pyrolite` for Total Alkali–Silica (TAS) classification and TAS-related visualization utilities. Users are encouraged to also cite:

> Williams, M. J., Schoneveld, L., Mao, Y., Klump, J., Gosses, J., Dalton, H., Bath, A., & Barnes, S. (2020). pyrolite: Python for geochemistry. *Journal of Open Source Software*, 5(50), 2314. https://doi.org/10.21105/joss.02314

This project uses `scikit-learn` for machine learning tools including ExtraTrees, one-class SVM, and nearest neighbors. Users are encouraged to also cite:

> Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in python. *Journal of Machine Learning Research*, 12(85), 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html

This project uses `SHAP` for feature-importance analysis. Users are encouraged to also cite:

> Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Proceedings of the 31st International Conference on Neural Information Processing Systems*, NIPS’17, 4768–4777.

Small portions of the liquid-mixing routines in this repository were adapted in part from `Thermobar`. If you use or adapt these routines, please also cite:

> Wieser, P., Petrelli, M., Lubbers, J., Wieser, E., Özaydın, S., Kent, A., & Till, C. (2022). Thermobar: An open-source Python3 tool for thermobarometry and hygrometry. *Volcanica*, 5(2), 349–384. https://doi.org/10.30909/vol.05.02.349384
