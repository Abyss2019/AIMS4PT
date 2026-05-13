# 给我用的备忘录 不要删！！
不提供eq test ，但是你可以用我们的代码


提醒用户不要轻易跑paper\notebooks里面的notebook，除非你知道你在干什么！
需要安装 pip install -e ".[dev]" 模式

因为里面有些notebook是用来复现AIMS4PT 的training的 (01_build_AIMS4PT_cpx_pressure_workflow.ipynb and 02_build_AIMS4PT_cpx_temperature_workflow.ipynb)，如果直接运行可能会覆盖掉之前的训练好的模型（可能会因为版本不同造成结果小幅波动？）。而且跑完所有的notebook可能需要很长时间，如果中断可能会导致数据不完整。

所以请务必先了解notebook的内容和目的，再决定是否运行。


忘了，我这个仓库里用了很小一部分 thermobar 的代码（但是没import 这个包） https://github.com/PennyWieser/Thermobar
还有引用了pyrolite package：  Williams et al., 2020 https://pyrolite.readthedocs.io/en/main/cite.html 
需要体现在我的readme 里面吗？


# AIMS4PT_cpx

This repository contains the code and supporting resources for **AIMS4PT_cpx**, an AI-assisted model-selection framework for clinopyroxene-based pressure-temperature (P-T) estimation. The framework evaluates the applicability of published clinopyroxene-only and clinopyroxene-liquid thermobarometers to a given dataset, runs P-T calculations, and exports model-selection and calculation reports.

## Repository Layout

- `calculator.ipynb`: User-friendly calculator notebook. This is the recommended entry point for applying AIMS4PT_cpx to new datasets.
- `input.xlsx`: Input template for `calculator.ipynb`.
  
- `src/aims4pt/`: Python source code for AIMS4PT. This includes thermobarometer models, out-of-distribution (OOD) and deviation prediction tools, data-processing utilities, visualization helpers, and Excel report generation.
- `src/aims4pt/model_tools/trained_model/`: Trained AIMS4PT_cpx resources, including OOD detectors, deviation functions, and SHAP results. Standard calculations use these files directly.

- `paper/`: Materials used for manuscript-related analyses, including public datasets, workflow notebooks.


## Installation

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

For routine application of the AIMS4PT_cpx framework described in the manuscript, use the root-level `calculator.ipynb` notebook:

1. Install the environment and package following the instructions above.
2. Enter sample data in `input.xlsx`. Clinopyroxene oxide columns should end with `_cpx`, liquid oxide columns should end with `_liq`, and compositions should be reported in wt%.
3. Open and run `calculator.ipynb`.
4. Check the cells marked as `Input required`, including whether clinopyroxene-liquid thermobarometers should be evaluated, the project name, and optional melt TAS information.
5. After execution, results are written to `report_output/<project_name>/`, including Excel reports and intermediate cache files.

Most users do not need to run the notebooks in `paper/notebooks/`.

## Development and Manuscript Reproduction

Use editable installation mode if you need to modify the AIMS4PT_cpx framework, recalibrate OOD/deviation/SHAP resources, or reproduce manuscript figures:

```bash
pip install -e ".[dev]"
```

**Important warning:** Do not run the notebooks in `paper/notebooks/` unless you understand which intermediate files and trained resources each notebook reads and writes. In particular, `02_build_AIMS4PT_cpx_temperature_workflow.ipynb` rebuilds the temperature workflow, and `03_reproduce_AIMS4PT_cpx_figures.ipynb` depends on outputs from the preceding workflow notebooks to reproduce manuscript figures. Running these notebooks directly may overwrite existing trained resources or intermediate products. `01_build_AIMS4PT_cpx_pressure_workflow.ipynb` is also a pressure-workflow rebuilding entry point and is not intended for routine use.

## Citation

If you use this repository or the AIMS4PT_cpx framework, please cite the manuscript:

> Liu, X.-Y., & Li, W.-R. (in preparation). Ranking Thermobarometers: An Artificial Intelligence-Based Framework for Selecting Clinopyroxene-Based Models.


