# Uncertainty analysis repeat workflow

本文档记录当前 `paper/notebooks/uncertainty_analysis/` 下不确定性实验的可重复流程。运行本项目命令时默认使用 AIMS4PT conda 环境：

```powershell
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe
```

不要用 base 环境或 Codex bundled Python 复现实验。

## 1. Analytical uncertainty

### 1.1 数据入口

- 主脚本：`paper/notebooks/uncertainty_analysis/run_uncertainty_experiments.py`
- 输入数据：`paper/data/independent_data_final.xlsx`
- Sheet：`Sheet1`
- 测试集筛选：`training/testing` 标准化为小写后等于 `testing` 或 `test`
- 当前 test subset 总数：59
- 当前 test subset 中 `H2O_liq != 0`：23；`H2O_liq == 0`：36

### 1.2 主分析方法

复现实验主要调用：

- CLI 实验名：`directional_equal_oat`
- Python 入口：`run_directional_equal_error_analysis(test_df, paths, reuse_existing)`
- 核心计算函数：`compute_directional_central_difference(test_df, specs, out_path)`
- 单元素扰动列表：`directional_single_oxide_specs(test_df, model_type)`
- 汇总函数：`summarize_directional_equal_error(long_df)`
- 作图函数：
  - `plot_directional_equal_error_figures(long_df, paths["analytical_fig"])`
  - `plot_directional_equal_abs_composites(long_df, paths["analytical_fig"])`

该方法是 deterministic one-at-a-time central plus/minus，不是 Monte Carlo。对每个 oxide 单独做 `+sigma` 和 `-sigma`，其他输入保持不变：

```text
sigma = original_value * relative_error
plus_value = original_value + sigma
minus_value = original_value - sigma
effect = (plus_prediction - minus_prediction) / 2
abs_effect = abs(effect)
```

因为当前全部使用相对误差，且相对误差均小于 100%，非负 oxide 的 minus perturbation 不需要再 clip 到 0。当前版本会把原始值为 0 的 oxide 标记为 `zero_value_skipped`，不纳入 median effect，避免由数据结构里的 0 值稀释敏感性。

### 1.3 Analytical uncertainty 参数

高含量 oxides 使用相对误差 +/-3%：

- `SiO2`
- `Al2O3`
- `FeO` / `FeOt`
- `MgO`
- `CaO`

低含量 oxides 使用相对误差 +/-8%：

- `TiO2`
- `MnO`
- `Na2O`
- `Cr2O3`
- `K2O`

水使用相对误差 +/-20%：

- `H2O_liq`

当前代码常量应保持为：

```python
CPX_TOTAL_MIN = 97.0
CPX_TOTAL_MAX = 103.0
CPX_STOICH_MIN = 0.9
CPX_STOICH_MAX = 1.1
DIRECTIONAL_RELATIVE_ERRORS = {
    "SiO2": 0.03,
    "TiO2": 0.08,
    "Al2O3": 0.03,
    "FeO": 0.03,
    "MgO": 0.03,
    "MnO": 0.08,
    "CaO": 0.03,
    "Na2O": 0.08,
    "Cr2O3": 0.08,
    "K2O": 0.08,
}
DIRECTIONAL_H2O_RELATIVE_ERROR = 0.20
```

QC 口径：

- cpx oxide total：97-103 wt.%
- cpx stoichiometry：`(Ca + Fe + Mg) / Si` 保持 0.9-1.1
- liquid total：不超过 105 wt.%
- 当前扰动不做 oxide total renormalization

### 1.4 cpx-only 中水的特殊处理

cpx-only 组也需要把 `H2O_liq` 作为可传入的 input，因为下列模型使用水：

- `Pu08_32b` 的 pressure prediction
- `Wan21` 的 temperature prediction

其他 cpx-only 模型不用人为加入水的响应。当前 runner 的处理方式是：

1. `directional_single_oxide_specs(test_df, "cpx_only")` 会额外生成 `phase="liq", storage_oxide="H2O_liq"` 的扰动 spec。
2. `model_uses_oxide(spec, "liq", "H2O_liq")` 检查该模型的 `standard_columns` 是否真的声明使用 `H2O_liq`。
3. 如果模型不使用该 oxide，则 plus/minus prediction 直接等于 baseline prediction，因此 effect 为 0。

因此，复现时不要删除 cpx-only 的 `H2O_liq` 列；但解释结果时只把 `Pu08_32b` 的 P 和 `Wan21` 的 T 当作真正需要水的 cpx-only case。

#### 模型信息来源

每个模型的更详细信息可以在 [model_tools](../../../src/aims4pt/model_tools/) 中查。复现实验时重点看各 wrapper 里的 `model_name`、`standard_columns`、`cpx_only`、`require_water`、`uncertainty` 和 `predict()` 实现，因为 analytical OAT 判断某个 oxide 是否真的被模型使用，主要依赖 `standard_columns`。

常用简称和源码入口：

| Short name | Source file | Notes for this workflow |
| --- | --- | --- |
| `AgL24` | `Agreda2024.py` | Ágreda-López et al. (2024) ML model；有 cpx-only 和 cpx-liq P/T 版本。 |
| `Chi23` | `Chicchi23.py` | Chicchi et al. (2023) GAIA/deep-learning model；有 cpx-only 和 cpx-liq P/T 版本。 |
| `Hig21` | `Higgins21.py` | Higgins et al. (2021) cpx-only P/T model。 |
| `Jor22` | `Jorgenson22.py` | Jorgenson et al. (2022) R-backed extraTrees model；有 cpx-only 和 cpx-liq P/T 版本。 |
| `Pet20` | `Petrelli20.py` | Petrelli et al. (2020) ML model；cpx-liq 版本的 `standard_columns` 包含 `H2O_liq`。 |
| `NP17` | `Neave_Putirka_17.py` | Neave & Putirka (2017) cpx-liq iterative barometer，当前图中主要是 P。 |
| `Pu08_31` / `Pu08_33` / `Pu08_32a` / `Pu08_32b` / `Pu08_32d` | `Putirka_08.py` | Putirka (2008) equation wrapper；`eq32b_P` 明确包含 `H2O_liq`，所以 `Pu08_32b` 的 P 要保留水输入。 |
| `Wan21` | `Wang_21.py` | Wang et al. (2021) cpx-only equations；`eq2_T` 明确包含 `H2O_liq`，所以 `Wan21` 的 T 要保留水输入。 |

模型池不是手写在 notebook 里，而是由 `run_uncertainty_experiments.py` 调用 `src/aims4pt/model_tools/model_registry.py` 的 `get_models_initial_pools()` 自动实例化。简称映射由 `paper/scripts/constants_illustration.py` 的 `get_model_abbreviation()` 控制；如果图上的简称和源码 class 名对不上，优先查这个映射函数。

### 1.5 推荐运行命令

在仓库根目录运行：

```powershell
cd "C:\Users\13493\Documents\PROJECTS\Cpx_thermobarometry_recommend_202507\code\AIMS4PT"
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe "paper\notebooks\uncertainty_analysis\run_uncertainty_experiments.py" --experiments directional_equal_oat
```

主数据输出保留，作为 Excel 可打开的表格输出：

- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/directional_equal_error_oat_long_test_subset.csv`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/directional_equal_error_oat_summary_test_subset.csv`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/directional_pm_abs_nonzero_oxide_counts_by_model_test_subset.csv`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/directional_pm_abs_nonzero_oxide_counts_by_oxide_test_subset.csv`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/directional_key_feature_boxes_nonzero_oxide_counts_test_subset.csv`

如果已有 long table 与当前 `DIRECTIONAL_SENSITIVITY_METHOD`、模型集合、行数和必需列兼容，脚本会复用旧结果并只重新汇总/作图。若要强制全量重算，先备份或移走旧的 `directional_equal_error_oat_long_test_subset.csv`。

主文最终图片是 3 × 2 combined heatmap，并且必须对应非零 oxide 筛选和 `H2O_liq` 20% relative error：

- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/figures_test_subset/analytical_uncertainty_maintext_combined.pdf`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/figures_test_subset/analytical_uncertainty_maintext_combined.png`

`test_subset_directional_pm_abs_composite_*_key_features.png` 和普通 composite 图可以作为中间过程或补充检查保留，但不作为主文最终展示图片。

### 1.6 非零 oxide 筛选后的样品量

当前 test subset 在按 oxide 非零值筛选后，需要报告以下样品量。样品量过低时，即使 median effect 看起来高，也要谨慎解释。

cpx oxides：

| Oxide | Nonzero sample n |
| --- | ---: |
| SiO2_cpx | 59 |
| TiO2_cpx | 59 |
| Al2O3_cpx | 59 |
| FeOt_cpx | 59 |
| MgO_cpx | 59 |
| CaO_cpx | 59 |
| MnO_cpx | 57 |
| Na2O_cpx | 54 |
| K2O_cpx | 42 |
| Cr2O3_cpx | 30 |

liquid oxides：

| Oxide | Nonzero sample n |
| --- | ---: |
| SiO2_liq | 59 |
| TiO2_liq | 59 |
| Al2O3_liq | 59 |
| FeOt_liq | 59 |
| MgO_liq | 59 |
| CaO_liq | 59 |
| Na2O_liq | 58 |
| MnO_liq | 56 |
| K2O_liq | 56 |
| H2O_liq | 23 |

重点 caveat：

- `H2O_liq` 只有 23 个非零样品，是最容易受样品量影响的变量。
- `Cr2O3_cpx` 只有 30 个非零样品，也需要谨慎解释。
- 其他 oxide 的非零样品量基本足够。

### 1.7 最终图片生成流程

当前主脚本已经把最终图片流程整合进 `directional_equal_oat`。正常情况下只需要运行 1.5 的主命令：

```powershell
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe "paper\notebooks\uncertainty_analysis\run_uncertainty_experiments.py" --experiments directional_equal_oat
```

主脚本会自动做四件事：

- 基于 `baseline_value != 0` 过滤每个 oxide 的作图统计，避免 0 值稀释 median effect。
- 写出非零 oxide 样品量统计表。
- 输出主文最终 `analytical_uncertainty_maintext_combined.pdf/png`。
- 同时保留 `_key_features.png` 红框补充图。

保留输出表格：

- `directional_pm_abs_nonzero_oxide_counts_by_model_test_subset.csv`
- `directional_pm_abs_nonzero_oxide_counts_by_oxide_test_subset.csv`
- `directional_key_feature_boxes_nonzero_oxide_counts_test_subset.csv`

主文 combined 图的主要设置：

- 3 行 × 2 列：row 1 为 cpx-only，row 2 为 cpx-liq 的 cpx perturbation，row 3 为 cpx-liq 的 liquid perturbation。
- 左侧三行标签依次为 `Clinopyroxene perturbation`、`Clinopyroxene perturbation`、`Liquid perturbation`。
- cpx-only barometry panel 保留 `H2O*` 主列；thermometry 中的 `H2O*`/`H2O` 从主 heatmap 中拆出，作为 panel b 和 panel f 右侧的独立窄 strip。
- panel b 和 panel f 的 H2O strip 使用独立于主 thermometry heatmap 的共享 H2O 色标；主 thermometry 色标只对应非 H2O 变量。H2O strip 和共享色标使用低饱和绿色系，主 heatmap 保持蓝色系。
- H2O 共享色标标题为 `H₂O effect on thermometers`。
- 灰色斜线方格表示该模型不使用该 feature；这和“使用但敏感性低”的浅蓝/白色格子区分开。
- 灰色斜线方格按 `run_uncertainty_experiments.py` 中的显式 main-text feature table 判定；如果表中该 phase 为 `N.A. in workbook`，则改用对应模型的 `model.standard_columns` 和 `require_water` 判定。
- 浅灰色但非 unused 的格子表示 QC 和非零 oxide 筛选后没有可用 median，不应解读为模型未使用该 feature。
- 主文 combined 图也保留 key features 的红色轮廓标识。
- `H2O_liq` 使用 20% relative error；不要再使用旧的 +/-3 wt.% absolute H2O 图。
- 主文 combined 图使用 matplotlib 绘制，`figsize=(9, 6)`；PNG/PDF/SVG 均通过 `fig.savefig(..., dpi=600)` 输出。
- 主文图内部不放 footnote、caption 或 Figure X 这类全局标题；这些解释放到 manuscript caption。

输出：

- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/figures_test_subset/analytical_uncertainty_maintext_combined.pdf`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/figures_test_subset/analytical_uncertainty_maintext_combined.png`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/figures_test_subset/analytical_uncertainty_maintext_combined.svg`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/figures_test_subset/test_subset_directional_pm_abs_composite_cpx_only_key_features.png`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/figures_test_subset/test_subset_directional_pm_abs_composite_cpx_liq_key_features.png`
- `paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/directional_key_feature_boxes_nonzero_oxide_counts_test_subset.csv`

如果 key feature 表以后再改，直接更新 `run_uncertainty_experiments.py` 中的 `directional_key_features()`。这会同时影响主文 combined 图和补充的红框 composite 图。`tmp/redraw_directional_pm_abs_nonzero_oxide.py` 和 `tmp/draw_directional_key_feature_boxes.py` 只是历史辅助脚本，不再是标准复现流程。

### 1.8 单独的 H2O 20% relative 表格检查

如果只需要复查水的影响，可以运行下面的表格检查。该步骤保留 CSV/Excel 兼容输出即可，不需要单独展开成最终图片：

```powershell
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe "tmp\run_h2o20_relative_directional_oat.py"
```

可选只跑某一种 case：

```powershell
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe "tmp\run_h2o20_relative_directional_oat.py" --case all
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe "tmp\run_h2o20_relative_directional_oat.py" --case h2o_nonzero
```

两个 case 的含义：

- `all`：所有 59 个 test subset 样品
- `h2o_nonzero`：只保留 `H2O_liq != 0` 的 23 个样品

输出目录：

```text
paper/notebooks/uncertainty_analysis/results/uncertainty/analytical/h2o20_relative/
```

当前结果解读口径：

- `all` case 会包含大量 `H2O_liq == 0` 的样品；这些样品对相对误差没有实际扰动，容易稀释 median。
- `h2o_nonzero` case 更适合解释真实含水样品中水对模型的影响。
- 对 cpx-only，重点看 `Pu08_32b` P 和 `Wan21` T；其他不声明使用水的模型不应被解释为有水敏感性。

### 1.9 已知坑

- 当前方法没有 Monte Carlo 分布，只是 deterministic central difference；不能直接解释成完整预测不确定度分布。
- 没有对扰动后的 oxide total 做 renormalization；结果代表单 oxide measurement error 的局部敏感性。
- 低含量或大量 0 值变量容易被样品结构影响；必须同时报告非零样品数。
- `H2O_liq` 样品量偏少，且只在部分模型中是真 input。
- cpx QC 从 98-102 wt.% 放宽到 97-103 wt.% 后，样品损失明显减少；复现时不要混用旧 QC。
- 如果 R-backed 模型报 `sh is not recognized`，说明当前 Windows/R shell 环境没有正确接上；这种情况下不要把失败行解释为地质或模型敏感性。

## 2. Kd uncertainty

### 2.1 数据入口

- 主脚本：`paper/notebooks/uncertainty_analysis/run_uncertainty_experiments.py`
- 输入数据：`paper/data/independent_data_final.xlsx`
- Sheet：`Sheet1`
- 测试集筛选：同第 1 节，`training/testing` 标准化为小写后等于 `testing` 或 `test`
- 依赖的 baseline 表：`baseline_cpx_liq_predictions_test_subset.csv`
- 当前 test subset 总数：59
- 当前有效 `Kd_FeMg` 数：59
- 当前 `Kd_FeMg` 在 0.20-0.36 之外的样品数：0

`kd` 实验不是重新配对 clinopyroxene-liquid，也不是改变 Merapi 应用中的 equilibrium Kd 值；它只在 independent test subset 的固定 cpx-liquid pair 上计算每个样品的 `Kd(Fe-Mg)`，然后检查模型 residual 是否随 Kd bin 系统变化。

### 2.2 主分析方法

复现实验主要调用：

- CLI 实验名：`kd`
- Python 入口：`run_kd_analysis(test_df, baseline, paths)`
- Kd 计算函数：`calculate_kd(test_df)`
- 回归诊断函数：`regression_metrics(x, y)`
- bin 统计函数：`kd_fine_bin_residual_iqr(merged, bin_edges)`
- 原始 runner 作图函数：`plot_kd_figures(...)`

Kd 计算使用 Fe-Mg 摩尔比：

```text
Fe_Mg_cpx = (FeO_cpx / 71.844) / (MgO_cpx / 40.304)
Fe_Mg_liq = (FeO_liq / 71.844) / (MgO_liq / 40.304)
Kd_FeMg = Fe_Mg_cpx / Fe_Mg_liq
Kd_centered = Kd_FeMg - 0.28
```

有效 Kd 需要 `FeO_cpx`、`MgO_cpx`、`FeO_liq`、`MgO_liq` 均可转为数值，并且 `MgO_cpx`、`FeO_liq`、`MgO_liq` 大于 0。无效行会标记为 `invalid FeO/MgO input`。

Residual 使用 baseline prediction 表里的：

```text
delta_P = P_pred - P_true
delta_T = T_pred - T_true
```

因此该实验回答的是：在固定 test subset 和固定 baseline 预测下，模型 residual 与样品的 `Kd(Fe-Mg)` 是否有关；它不等价于完整的 Kd pairing uncertainty propagation。

### 2.3 Kd bin 和统计口径

当前最终 supp 图和复现口径只使用 0.02 宽度 bin。代码中应保持的 bin 常量为：

```python
KD_FINE_BIN_EDGES = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36], dtype=float)
```

`calculate_kd()` 内部还会生成较粗的解释性 `Kd_bin`，可用于中间检查，但不要作为最终 Figure SXX 的 bin 口径：

```text
outside_0.20_0.36
0.20_0.24
0.24_0.28
0.28_0.32
0.32_0.36
outside_0.20_0.36
```

`kd_bin_residual_iqr_0p02_by_model_test_subset.csv` 和最终补充图使用 0.02 宽度 bin。每个 model-target-bin 保留：

- `median_residual`
- `residual_q1`
- `residual_q3`
- `mean_residual`
- `rmse`

注意：最终 residual distribution 图中的 vertical bars 是 `residual_q1` 到 `residual_q3`，也就是 interquartile interval (Q1-Q3)，不是 standard deviation。当前重画脚本用的是：

```text
lower = median_residual - residual_q1
upper = residual_q3 - median_residual
```

如果以后要画 std，必须回到样品级 residual 重新按 model-target-bin 计算 `mean ± SD` 或 `median ± SD`，不能直接用当前 IQR CSV 替代。

### 2.4 当前 Kd bin 样品量

0.02 bin：

| Kd bin | Center | Unique sample n |
| --- | ---: | ---: |
| 0.20-0.22 | 0.21 | 4 |
| 0.22-0.24 | 0.23 | 19 |
| 0.24-0.26 | 0.25 | 10 |
| 0.26-0.28 | 0.27 | 8 |
| 0.28-0.30 | 0.29 | 9 |
| 0.30-0.32 | 0.31 | 5 |
| 0.32-0.34 | 0.33 | 2 |
| 0.34-0.36 | 0.35 | 2 |

解释时要特别注意高 Kd 端样品量偏低：0.32-0.34 和 0.34-0.36 在 0.02 bin 中各只有 2 个样品。这些 bin 可以帮助显示趋势，但不适合过度解释为稳健统计结论。

### 2.5 推荐运行命令

在仓库根目录运行：

```powershell
cd "C:\Users\13493\Documents\PROJECTS\Cpx_thermobarometry_recommend_202507\code\AIMS4PT"
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe "paper\notebooks\uncertainty_analysis\run_uncertainty_experiments.py" --experiments kd
```

`kd` 会自动需要 cpx-liquid baseline prediction。默认 `--reuse-existing` 为开启状态；如果找到兼容的 baseline 表就复用，否则会重新计算并写出：

- `paper/notebooks/uncertainty_analysis/results/uncertainty/baseline_cpx_liq_predictions_test_subset.csv`

主要 Kd 输出保存在：

```text
paper/notebooks/uncertainty_analysis/results/uncertainty/kd/
```

需要保留的表格输出：

- `kd_values_by_sample_test_subset.csv`
- `kd_regression_by_model_test_subset.csv`
- `kd_bin_summary_by_model_test_subset.csv`
- `kd_bin_residual_iqr_0p02_by_model_test_subset.csv`
- `kd_bin_counts_0p02_test_subset.csv`
- `kd_analysis_log_test_subset.csv`

当前复现只保留 0.02 signed residual 的中间检查图：

- `paper/notebooks/uncertainty_analysis/results/uncertainty/kd/figures_test_subset/test_subset_kd_bin_residual_iqr_composite.png`


### 2.6 最终 supp 图重画流程

当前 supp 中使用重画后的 `Figure SXX`，不是 runner 自动生成的 PIL 原图。重画脚本为：

```text
tmp/redraw_kd_bin_residuals.py
```

该脚本读取：

```text
paper/notebooks/uncertainty_analysis/results/uncertainty/kd/kd_bin_residual_iqr_0p02_by_model_test_subset.csv
```

输出：

- `paper/.images/final_supp/Figure_SXX_Kd_bin_residual_distributions.png`
- `paper/.images/final_supp/Figure_SXX_Kd_bin_residual_distributions.pdf`
- `paper/.images/final_supp/Figure_SXX_Kd_bin_residual_distributions.svg`

重画版本的图面规范：

- 不放总标题。
- 不放解释性副标题。
- 不画 `Kd = 0.28` 垂直参考线。
- 不在图上标 `n=` 样品数。
- 只保留 Pressure / Temperature 面板标识。
- 保留 dashed horizontal zero-residual line。
- vertical bars 为 Q1-Q3，不是 std。
- 保持 6.0 inch 宽、约 3.817 inch 高，方便替换到 supp Word 中原 Figure SXX 尺寸。

生成最终图：

```powershell
C:\Users\13493\miniconda3\envs\AIMS4PT\python.exe "tmp\redraw_kd_bin_residuals.py" --plot-only
```

如果还需要同步生成替换图片和图注后的 Word 副本，使用带 `python-docx` 的 Python 环境运行 `--docx-only`。当前 Codex bundled Python 可用：

```powershell
C:\Users\13493\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe "tmp\redraw_kd_bin_residuals.py" --docx-only
```

输出 Word 副本：

- `output/doc/AGU_SuppInfo_Word_xy_0705_kd_redrawn.docx`

当前聊天中还把该副本另存到了 MyPaper 目录，但不覆盖原始 `AGU_SuppInfo_Word_xy_0705.docx`：

- `C:/Users/13493/Documents/PROJECTS/Cpx_thermobarometry_recommend_202507/MyPaper/AGU_SuppInfo_Word_xy_0705_kd_redrawn.docx`

### 2.7 图注口径

最终 supp 图注应说明：

- bins span `Kd(Fe-Mg) = 0.20-0.36` at 0.02 intervals；
- points show median residual；
- vertical bars show interquartile ranges / interquartile interval `(Q1-Q3)`；
- residual 定义为 predicted minus experimental pressure or temperature；
- dashed horizontal line marks zero residual；
- panel (a) pressure，panel (b) temperature；
- model abbreviations follow Table 1。

最终图没有 `n=` labels，也没有 `Kd = 0.28` vertical line，因此图注不要再写：

- Grey labels indicate sample counts；
- vertical dashed line marks Kd = 0.28。

### 2.8 已知坑

- 这个实验不是 Monte Carlo，也不是完整的 Kd-pairing perturbation；它只是固定 test pair 上的 Kd-residual association analysis。
- 当前最终图的误差棒是 Q1-Q3，不是 std；不要在正文或图注里写成 standard deviation。
- 0.32-0.36 的高 Kd 端样品量很少，尤其 0.02 bin 中每个 bin 只有 2 个样品，解释趋势时必须保守。
- `kd_regression_by_model_test_subset.csv` 中的 slope 是基于 `Kd_centered = Kd_FeMg - 0.28`，其中 `slope_per_0.01Kd = slope_per_1Kd * 0.01`。
- `plot_kd_figures()` 里的 scatter、slope summary、RMSE lineplot 代码是不可达的旧代码，不要把它们当成当前复现流程的一部分。
- 如果重画 supp 图后更改了图面元素，必须同步更新 Figure SXX 图注，避免图注描述旧图里的 n labels 或 Kd=0.28 reference line。
