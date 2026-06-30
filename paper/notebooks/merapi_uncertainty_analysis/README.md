# Merapi uncertainty analysis

This folder contains the Merapi version of the analytical OAT sensitivity workflow.

Run:

```powershell
conda run -n AIMS4PT python paper/notebooks/merapi_uncertainty_analysis/run_merapi_analytical_oat.py
```

Default input is the cached kd=0.28 Merapi cpx-liquid pairing output in:

```text
paper/.cache/add_pre-2006_028/
```

Default output is:

```text
paper/notebooks/merapi_uncertainty_analysis/results/merapi_kd028_pass/
```
