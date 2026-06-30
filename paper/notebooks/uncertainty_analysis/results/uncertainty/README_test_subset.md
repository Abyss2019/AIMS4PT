# Test-subset uncertainty analysis

This folder contains uncertainty and sensitivity outputs generated from only the independent experimental test subset.

- Input file: `paper/data/independent_data_final.xlsx`, sheet `Sheet1`.
- Split filter: rows where `training/testing` is `testing` or `test` after case-insensitive normalization.
- Retained test-subset rows: 59.
- Total input rows: 293; training rows: 234.
- Core baseline/Kd/analytical OAT models: cpx-liquid thermobarometers only; cpx-only models were excluded from those core outputs.
- Directional equal-error OAT extension: cpx-liquid and cpx-only model groups are plotted separately at matched 1% and 2% positive perturbations.
- Pairing: fixed experimental cpx-liquid pairs were used; liquids were not re-paired.
- Kd analysis: association between observed Kd(Fe-Mg) and model predictions/residuals, not a re-pairing experiment.
- Analytical perturbations: cpx oxides at 0.5% and 1%; liquid oxides at 1%, 2%, and 3%.
- Perturbation method: deterministic one-at-a-time plus/minus relative perturbations; no Monte Carlo.
- Cpx QC: perturbed cpx compositions must pass 98-102 wt.% total and 0.9-1.1 stoichiometric ratio before model calculation.
- Liquid QC: perturbed liquid compositions must pass the liquid total upper-bound check of <=105 wt.% before model calculation.
- H2O handling: H2O columns were kept fixed and were not perturbed.
- Baseline reuse: C:\Users\13493\Documents\PROJECTS\Cpx_thermobarometry_recommend_202507\code\AIMS4PT\paper\notebooks\uncertainty_analysis\results\uncertainty\baseline_cpx_liq_predictions_test_subset.csv
