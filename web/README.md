# AIMS4PT Web Calculator

This directory contains a Streamlit interface for the root-level `calculator.ipynb`
workflow. It reads an AIMS4PT Excel input file, runs the applicable
clinopyroxene-only and clinopyroxene-liquid workflows, and writes Excel reports to
`report_output/<project_name>/`.

The web app only applies existing packaged model resources. It does not train
models or recompute SHAP results.

## Installation

From the repository root:

```bash
pip install -e ".[web]"
```

If the package is already installed, installing Streamlit is enough:

```bash
pip install streamlit
```

## Run

From the repository root:

```bash
streamlit run web/app.py
```

The app can use the bundled `input.xlsx` or an uploaded `.xlsx` file. Input column
rules match `calculator.ipynb`: clinopyroxene oxide columns should end with
`_cpx`, liquid oxide columns should end with `_liq`, and compositions should be
reported in wt%. The web interface currently limits each input file to 500 rows.

Excel parsing is cached with `st.cache_data`, and AIMS4PT model imports/model-pool
initialization are cached with `st.cache_resource`, so normal Streamlit reruns do
not reload all models.

## Outputs

The app writes reports and workflow caches to:

```text
report_output/<project_name>/
```

Generated Excel reports keep the same workbook structure as the notebook reports:
`Results`, `Model Summary`, and `Ranking Details`.
