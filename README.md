# Auditable water-balance forecasting of civil water demand and network losses — ATO1 Marche Nord (Italy, 2017–2021)

Code and reference outputs accompanying the manuscript:

> **Auditable territorial water balances and honestly-benchmarked deep learning for water-security forecasting: a case study of ATO1 Marche Nord (Italy).**
> Nicolò Ciuccoli¹, Jessica Caroline dos Santos Silva², Franciele Maria Vanelli², Leonardo Romero Monteiro², Davide Fronzi¹, Anna Laura Eusebi¹, Virginia Grace Barros²\*, Francesco Fatone¹
>
> ¹ Department of Science and Engineering of Materials, Environment and Urban Planning (SIMAU), Università Politecnica delle Marche, Ancona, Italy
> ² Water Sciences Lab (LaCia), Civil Engineering Department, Santa Catarina State University (UDESC), Joinville, Santa Catarina, Brazil
> \* Corresponding author: virginia.barros@udesc.br
>
> Submitted to *Environmental Modelling & Software* (DOI to be added on publication).

This repository contains the three modelling pipelines (Jupyter notebooks, with their executed outputs), the cached public covariates, the per-seed and aggregated metric tables from which the manuscript's tables and skill figures are derived, and the plotting script for the manuscript figures. **The operator-supplied monthly series are not redistributed.** The same applies to the model-prediction tables, which embed the observed values. Both are available on request; see [§2](#2-study-area-and-data) and [`data/README_DATA.md`](data/README_DATA.md).

---

## 1. Overview

The study couples a *territorially explicit, auditable monthly water balance* for the ATO1 Marche Nord integrated water-service area (Province of Pesaro-Urbino, Marche, central Italy) with data-driven forecasting of two management-relevant variables: **civil water demand** (CWD; *Fabbisogni*) and **network losses** (NL; *Perdite*). The record covers 60 months (January 2017 – December 2021).

The central methodological question is not "which model is best". It is "*how much value does model complexity add on the short, seasonal records that most utilities actually hold, when the comparison is made honestly*". Several model families are compared:

- deep-learning architectures (LSTM, Transformer, CNN-LSTM) and a multilayer perceptron;
- gradient-boosted trees (XGBoost);
- a statistical baseline (SARIMA);
- zero-parameter naïve baselines (persistence, seasonal-naïve).

All are evaluated under a **leakage-free rolling-origin (walk-forward) protocol with five-seed reporting**, across three feature specifications of increasing dimensionality.

The headline results are deliberately sobering:

- **Demand.** A zero-parameter seasonal-naïve rule is not robustly surpassed by any trained model. XGBoost exceeds it only marginally, at 5–6 months ahead in the long-horizon experiments (ΔNSE ≤ +0.05).
- **Losses.** No method beats the observed mean (NSE = 0).
- **Covariates.** Adding hydroclimatic and tourism covariates *degrades* out-of-sample skill.

These findings motivate the recommended parsimonious specification.

---

## 2. Study area and data

**Domain.** ATO1 Marche Nord ("Ambito Territoriale Ottimale" No. 1) is the integrated water-service area of the Province of Pesaro-Urbino, on the Adriatic flank of the Central Apennines. It is served by Marche Multiservizi S.p.A. (MMS) and ASET S.p.A. Over 2017–2021 the supply side of the balance comprises:

| Supply term | Share of 2017–2021 supply |
|---|---|
| Treated drinking water from potabilization plants (*Potabilizzatori*) | ≈ 63% |
| Well-field abstraction (*Pozzi*) | ≈ 20% |
| Spring abstraction (*Sorgenti*) | ≈ 17% |

**Temporal coverage.** 60 consecutive months, January 2017 – December 2021, at monthly resolution. All volumetric quantities are in m³·month⁻¹.

**Water-balance identity.** Every monthly volume is traceable to an institutional or operator source:

```
Sorgenti + Pozzi + Potabilizzatori  =  Fabbisogni + Perdite
```

Network losses (*Perdite*) are obtained as the balance residual.

**Variables and provenance.**

| Symbol | Description | Role | Source |
|---|---|---|---|
| `Sorgenti` | Spring abstraction | driver | MMS / ASET operational records |
| `Pozzi` | Well-field abstraction | driver | MMS / ASET operational records |
| `Popolazione` | Served (resident) population | driver | ISTAT resident-population statistics (annual, held constant within year) |
| `Potabilizzatori` | Treated drinking-water (potabilization) volume | driver | MMS / ASET potabilization-plant records |
| `Fabbisogni` | Civil water demand | **target** | Metered consumption (MMS / ASET) |
| `Perdite` | Network losses | **target** | Derived balance residual (supply − demand) |
| `temp_C`, `precip_mm`, `pet_era5_mm` | Air temperature, precipitation, potential evapotranspiration | extended covariate | ERA5-Land monthly reanalysis (Muñoz-Sabater et al., 2021) |
| `SPI_3`, `SPEI_3` | 3-month standardized precipitation / precipitation-evapotranspiration indices | extended covariate | Derived from ERA5-Land; distributions fitted on the 1991–2020 reference period and scored over 2017–2021 (McKee et al., 1993; Vicente-Serrano et al., 2010) |
| `tourist_pop_equiv` | Monthly tourist-equivalent population | extended covariate | ISTAT, *Movimento dei clienti negli esercizi ricettivi* (monthly overnight stays, Marche region, NUTS-2 code ITE3) |
| `seismic_flag`, `seismic_decay` | Post-seismic step and exponential-decay term (τ = 18 months) | parsimonious/extended covariate | 2016 Amatrice (Mw 6.0, 24 Aug 2016) – Norcia (Mw 6.5, 30 Oct 2016) sequence (Chiaraluce et al., 2017) |

The seismic decay term encodes the fading post-seismic perturbation of spring discharge in the source aquifers. It is strongest in early 2017 (decay ≈ 0.9) and falls to ≈ 0.06 by 2021.

**Cached public data** (`data_cache/`) let the extended pipeline run without Google Earth Engine or I.Stat access:

- `era5_land_monthly.csv`: ERA5-Land monthly series over the study period;
- `era5_land_climatology_1991_2020.csv`: the 1991–2020 ERA5-Land record (360 months), used only to fit the SPI/SPEI distributions;
- `istat_tourism.csv`: monthly ISTAT overnight-stay (*presenze*) series for the Marche region (ITE3).

### Data availability

The operator-supplied monthly series were provided by MMS and ASET and are subject to their data-sharing conditions. They are spring and well abstraction, potabilized volume and metered demand, together with the losses derived from them. They are **not included** in this repository. The following files are therefore withheld (and listed in `.gitignore`):

| Withheld file | Content |
|---|---|
| `data/Dataset.xlsx` | Monthly operational series, 2017–2021 |
| `output_configc/configc_predictions_long.csv` | Per-origin observed/predicted values, parsimonious pipeline |
| `output_part2_5seed/configa_extended_predictions_long.csv` | Per-origin observed/predicted values, extended pipeline |
| `output_part2_5seed/integrated_dataset.csv` | Operational series merged with the covariates (written by the extended notebook) |

The processed series can be made available by the corresponding author on reasonable request and with the permission of the data providers. Notebook cells that displayed rows of these series have had that output removed (a note marks each one); all other executed outputs are kept as run.

ERA5-Land data: *Contains modified Copernicus Climate Change Service information (1991–2021)*. Neither the European Commission nor ECMWF is responsible for any use of this information. ISTAT statistics: © Istituto Nazionale di Statistica.

---

## 3. Repository structure

```
ATO1-Marche-Nord-forecasting/
├── README.md
├── LICENSE                               # MIT (code); data are not covered — see §2
├── CITATION.cff
├── requirements.txt                      # Python dependencies (see §7)
├── environment/                          # pip-freeze snapshots of the two environments used (see §7)
├── .gitignore
│
├── notebooks/
│   ├── ATO1_minimal_pipeline.ipynb       # minimal specification (7 features)
│   ├── ATO1_extended_pipeline.ipynb      # extended specification (14 features) + feature-group ablation
│   └── ATO1_parsimonious_pipeline.ipynb  # parsimonious specification (9 features)  [RECOMMENDED]
│
├── figures/
│   └── make_figures.py                   # regenerates manuscript Figs. 2–13 (see §8)
├── analysis/
│   └── dm_tests_vs_seasonal_naive.py     # Diebold–Mariano tests reported in Section 3.2 (see §8)
│
├── data/
│   └── README_DATA.md                    # data availability (Dataset.xlsx is withheld)
├── data_cache/                           # cached public covariates (used by the extended pipeline)
│   ├── era5_land_monthly.csv
│   ├── era5_land_climatology_1991_2020.csv
│   └── istat_tourism.csv
│
├── output_v2/                            # reference outputs — minimal pipeline
│   ├── master_metrics_v2_agg.csv
│   └── master_metrics_v2_per_seed.csv
├── output_configc/                       # reference outputs — parsimonious pipeline (manuscript headline)
│   ├── configc_all_metrics_agg_5seed.csv       # NSE, KGE (+ r, α, β), PBIAS, MASE, RMSE, MAE, sMAPE: mean ± SD
│   ├── configc_all_metrics_per_seed_5seed.csv
│   ├── master_metrics_configc_agg.csv          # NSE / KGE mean ± SD by model × horizon × target × experiment
│   └── master_metrics_configc_per_seed.csv
└── output_part2_5seed/                   # reference outputs — extended pipeline
    ├── ablation_final.csv                      # feature-group ablation (LSTM, demand/losses, 5 seeds)
    ├── master_metrics_part2_agg_final.csv
    └── master_metrics_part2_per_seed_final.csv
```

All paths in the notebooks are **relative to the repository root**. The first code cell of each notebook moves the working directory to the root if the kernel starts inside `notebooks/`. The notebooks therefore run without path edits in Jupyter or VS Code, provided `data/Dataset.xlsx` is present.

---

## 4. Feature specifications

All three specifications share the same evaluation framework and differ only in the predictor set. This lets the *marginal value of added information* be measured rather than assumed.

| Specification | n | Composition | External data | Notebook | Internal label / output folder |
|---|---|---|---|---|---|
| **Minimal** | 7 | 4 operational drivers + cyclical calendar (month sin/cos, year index) | none | `ATO1_minimal_pipeline.ipynb` | "v2" → `output_v2/` |
| **Parsimonious** *(recommended)* | 9 | minimal + 2 seismic covariates | none | `ATO1_parsimonious_pipeline.ipynb` | "Config C" → `output_configc/` |
| **Extended** | 14 | minimal + hydroclimatic (temperature, precipitation anomaly, SPI-3, SPEI-3) + tourist-equivalent population + 2 seismic covariates | ERA5-Land; ISTAT | `ATO1_extended_pipeline.ipynb` | "Config A" / "Part 2" → `output_part2_5seed/` |

The parsimonious 9-feature set was adopted *a posteriori*, on the basis of the extended set's own feature-group ablation. That ablation showed that removing the hydroclimatic block produced the single largest improvement in out-of-sample skill (`output_part2_5seed/ablation_final.csv`).

---

## 5. Notebooks

Each notebook is self-contained and reproduces one feature specification end-to-end:

- data loading and feature engineering;
- leakage-free rolling-origin backtesting across all model families and five seeds;
- metric computation and Diebold–Mariano significance testing;
- export of the prediction and metric tables.

All three require `data/Dataset.xlsx`, which is withheld (see §2).

**`ATO1_minimal_pipeline.ipynb`: minimal (7-feature) baseline.** Uses the four operational drivers plus a cyclical calendar encoding. Models: persistence, seasonal-naïve, SARIMA, LSTM, Transformer. Writes to `output_v2/`.

**`ATO1_extended_pipeline.ipynb`: extended (14-feature) analysis.** Adds the ERA5-Land hydroclimatic block, the SPI-3/SPEI-3 drought indices, the ISTAT tourist-equivalent population and the two seismic covariates. It runs from the cached `data_cache/` files: the Earth Engine and I.Stat fetch cells are used only if the cache is missing, and need your own Earth Engine project id (`gee_project`). Models: the minimal set plus XGBoost and CNN-LSTM. The notebook includes the feature-group ablation and single-run SHAP and MC-Dropout diagnostics. Writes to `output_part2_5seed/`.

**`ATO1_parsimonious_pipeline.ipynb`: parsimonious (9-feature) specification (recommended).** This is the manuscript's headline pipeline: the four drivers, the calendar encoding and the two seismic covariates, with no external data. Models: the extended set plus a multilayer perceptron. It produces the metric tables behind the article's performance tables and the in-notebook Diebold–Mariano comparisons between model families. The tests against seasonal-naïve reported in Section 3.2 are reproduced by `analysis/dm_tests_vs_seasonal_naive.py`. Writes to `output_configc/`.

> **Multi-seed protocol.** Every neural model has stochastic weight initialization and XGBoost has stochastic subsampling, so a single run is one draw from a distribution rather than a definitive score. All three pipelines are evaluated across **five random seeds (42–46)**, and every reported metric is a mean ± standard deviation (sample SD) across seeds. Deterministic families (persistence, seasonal-naïve, SARIMA) are re-run at every seed as a self-consistency check; their across-seed SD is ≈ 0.

---

## 6. Methodology

**Leakage-free rolling-origin (walk-forward) evaluation.** At each origin, feature scalers are fitted on the training partition only and applied forward. Models are trained on data strictly preceding the origin and scored on the held-out horizon. Short-horizon models (static mode) are trained once and rolled forward; long-horizon models (expanding mode) are retrained at every origin.

**Forecast configurations.** There are two horizons: a short *operational* horizon (3 months in → 2 months out) and a long *strategic* horizon (12 → 12). Each is run under two input schemes, which appear in the output tables as the `experiment` column:

| `experiment` value | Inputs |
|---|---|
| `short_term_multivariate`, `long_term_multivariate` | The specification's features only: operational drivers, calendar and, where applicable, seismic and external covariates. No target history. |
| `short_term_autoregressive`, `long_term_autoregressive` | The same features plus one-month lags of both targets |

The article's performance tables (Tables 4–5) and Figs. 5–13 report the `*_multivariate` experiments.

**Model families.**

| Family | Configuration |
|---|---|
| Naïve baselines | Persistence; seasonal-naïve |
| SARIMA | statsmodels `SARIMAX`, with a magnitude guard rejecting explosive non-stationary forecasts |
| XGBoost | 200 trees, depth 3, learning rate 0.05, subsample / colsample 0.8 |
| LSTM (stacked) | 64 → 32 units, 32-unit dense head, dropout 0.3, L2 = 1e-4, early stopping |
| Transformer | 2 encoder blocks, 4 attention heads, feed-forward width 64, sinusoidal positional encoding |
| CNN-LSTM | Conv1D (32 filters) → LSTM (48 units) → dense head |
| MLP (parsimonious pipeline) | Dense 32 → 16 |

**Metrics.** The headline measures are the Nash–Sutcliffe Efficiency (NSE; NSE = 0 corresponds to predicting the observed mean) and the Kling–Gupta Efficiency (KGE, with its correlation/bias/variability decomposition). These are complemented by percent bias (PBIAS), the mean absolute scaled error (MASE), sMAPE, RMSE and MAE.

**Significance testing.** Pairwise differences in predictive accuracy are assessed with the Diebold–Mariano test. It uses a squared-error loss and the Harvey–Leybourne–Newbold small-sample correction, and is run independently within each seed.

---

## 7. Reproducing the results

**Environments.** The reference outputs were produced on Windows (Anaconda, VS Code) with two Python environments, as recorded in each notebook's metadata:

| Notebook | Python | Environment |
|---|---|---|
| `ATO1_minimal_pipeline.ipynb` | 3.11.15 | conda env `neural311` |
| `ATO1_extended_pipeline.ipynb`, `ATO1_parsimonious_pipeline.ipynb` | 3.13.5 | Anaconda `base` |

The exact package versions of both environments are recorded as `pip freeze` snapshots in `environment/`. `requirements.txt` lists the packages without version pins. For an exact match, install from the snapshot corresponding to the notebook you run:

```bash
pip install -r requirements.txt                                     # quick start
pip install -r environment/requirements-lock-py313.txt              # exact: extended / parsimonious
pip install -r environment/requirements-lock-py311.txt              # exact: minimal
```

The parsimonious and extended notebooks install `xgboost` automatically if it is missing. `shap` and `earthengine-api` are optional: they are used only for the extended notebook's SHAP diagnostic and the online data fetch.

**Run order.** The notebooks are independent; the intended narrative order is minimal → extended → parsimonious. Place `data/Dataset.xlsx` in `data/`, open each notebook in `notebooks/` with Jupyter or VS Code, and *Run All*.

**Runtime.** Runtime is dominated by the neural models and grows linearly with `N_SEEDS`. The extended notebook takes on the order of 2–3 h at five seeds on a laptop CPU, because long-horizon models retrain at every rolling origin.

**Determinism.** Re-running with the shipped seeds (42–46) should reproduce the reference metric tables in `output_*` up to minor last-decimal differences in the neural models. Such differences are expected across hardware, BLAS and TensorFlow builds, and are why results are reported as a mean ± SD over seeds.

---

## 8. Mapping of outputs to the manuscript

| Manuscript element | Source file(s) | Generated by |
|---|---|---|
| Tables 4–5 (CWD / NL NSE & KGE; short = mean of h = 1–2, long = mean of h = 1–12; `*_multivariate` experiments) | `output_configc/master_metrics_configc_agg.csv` | parsimonious notebook |
| Complete metric set (PBIAS, MASE, sMAPE, RMSE, MAE, KGE components) | `output_configc/configc_all_metrics_*_5seed.csv` | parsimonious notebook |
| Figs. 5, 6, 10 (skill by horizon, long-term multivariate) | `output_configc/master_metrics_configc_agg.csv` | `figures/make_figures.py` |
| Fig. 13 (seed dispersion, demand, h = 1) | `output_configc/master_metrics_configc_agg.csv` | `figures/make_figures.py` |
| Fig. 12 (feature parsimony) | `output_v2/master_metrics_v2_agg.csv`, `output_configc/master_metrics_configc_agg.csv`, `output_part2_5seed/master_metrics_part2_agg_final.csv` | `figures/make_figures.py` |
| Feature-group ablation (text) | `output_part2_5seed/ablation_final.csv` | extended notebook |
| Figs. 7, 8 (observed vs modelled); Figs. 9, 11 (hindcasts) | `configc_predictions_long.csv` (withheld) + `master_metrics_configc_agg.csv`; Figs. 9/11 also need `Dataset.xlsx` (withheld) | `figures/make_figures.py` |
| Figs. 2, 3 (raw series; month-of-year profile) | `data/Dataset.xlsx` (withheld) | `figures/make_figures.py` |
| Fig. 4 (forecast-design schematic) | — | `figures/make_figures.py` |
| Diebold–Mariano tests vs seasonal-naïve (Section 3.2) | `configc_predictions_long.csv` (withheld) | `analysis/dm_tests_vs_seasonal_naive.py` |
| Diebold–Mariano tests between model families (MLP vs LSTM, LSTM vs Transformer) | computed in-notebook; results printed in the executed notebooks | parsimonious / minimal notebooks |
| Fig. 1 (study-area map) | produced in GIS; not generated by this code | — |

```bash
python figures/make_figures.py              # writes figures/output/FigureNN_*.png (300 dpi) and .pdf
python figures/make_figures.py --titles     # same, with figure-level titles and footnotes drawn in
python analysis/dm_tests_vs_seasonal_naive.py   # Diebold–Mariano tests of Section 3.2 (needs the withheld predictions)
```

By default the figures are drawn as submitted with the article: without figure-level titles and footnotes, because that information is in the captions. Figures whose withheld inputs are absent are skipped with a message. With the public files alone, Figs. 4, 5, 6, 10, 12 and 13 are regenerated. The prediction tables follow a common tidy schema: `origin_idx, origin_date, target_date, horizon, target, model, obs, pred, experiment, seed`.

---

## 9. Key findings

1. **Skill depends on the target and horizon far more than on model sophistication.** Civil demand is strongly predictable; network losses are not.
2. **A zero-parameter seasonal-naïve rule is not robustly surpassed for CWD.** Under the recommended specification, XGBoost exceeds it only marginally, at 5–6 months ahead in the long-horizon experiments (ΔNSE ≤ +0.05; e.g. 0.72 vs 0.68 at h = 5, multivariate scheme). Gradient-boosted trees and the LSTM otherwise match but do not exceed it. The Transformer, CNN-LSTM and MLP fall far below it.
3. **Network losses are not predictable with the available information.** Under the parsimonious and minimal specifications, NSE is below zero for every model, horizon and experiment. The only positive value in any specification is the extended-specification LSTM at h = 1 (0.10 ± 0.21), which is not distinguishable from zero. This is an information-limited ceiling, consistent with losses being a balance residual and an event-driven process rather than a smooth function of the available covariates.
4. **Parsimony wins.** Adding hydroclimatic and tourism covariates degrades out-of-sample skill (LSTM demand NSE at h = 1: 0.66 ± 0.04 with 9 features vs 0.10 ± 0.23 with 14). Those variables largely restate the seasonal position the calendar already encodes, and with only 60 observations redundant collinear features add more overfitting risk than signal.

---

## 10. How to cite

Please cite the article (see also `CITATION.cff`):

```
Ciuccoli, N., dos Santos Silva, J.C., Vanelli, F.M., Monteiro, L.R., Fronzi, D., Eusebi, A.L.,
Barros, V.G., Fatone, F., [year]. Auditable territorial water balances and honestly-benchmarked
deep learning for water-security forecasting: a case study of ATO1 Marche Nord (Italy).
Environmental Modelling & Software, [volume], [article number]. [DOI — to be added on publication]
```

A version-tagged release of this repository will be archived with a citable DOI (via Zenodo) upon acceptance.

---

## 11. License

The **code** is released under the MIT License (see `LICENSE`). The **operator data** are not part of this repository and are not covered by that license. The cached ERA5-Land- and ISTAT-derived files in `data_cache/` remain subject to their providers' terms (see §2).

---

## 12. Contact

Virginia Grace Barros — Water Sciences Lab (LaCia), Civil Engineering Department, Santa Catarina State University (UDESC), Joinville, Brazil — virginia.barros@udesc.br

---

## 13. Note on the use of AI-assisted tools

Generative-AI assistance was used to support the review, debugging and documentation of the analysis code and of this repository. The study design, all model training and evaluation, and all reported results were produced and verified by the authors using the code herein. No generative-AI tool was used to generate, collect, alter or interpret the underlying data. All figures derive from the authors' plotting code applied to the numerical outputs above.
