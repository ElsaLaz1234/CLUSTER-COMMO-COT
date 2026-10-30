# CLUSTER-COMMO-COT
# Do COT Positioning Clusters Anticipate Commodity Return Co-movement?

A cross-sectional clustering study that tests whether managed money positioning, as reported in the CFTC Commitments of Traders (COT), anticipates which commodity futures will move together at 1–3 month horizons. Hierarchical clustering with Ward linkage is applied each week to both COT positioning and realized returns across 16 commodity futures from 2006 to 2024, and the two partitions are compared with the Adjusted Rand Index (ARI).

---

## Motivation

The CFTC COT report is one of the few publicly available sources of institutional positioning data with weekly frequency and a clean trader category decomposition. Commodity futures markets are structurally bipartite — commercial hedgers hedge physical exposure while managed money (hedge funds, CTAs) speculate on price direction. This separation gives COT a direct economic interpretation.

The question is not whether positioning predicts individual commodity returns — Wang & Zhang (2024) study individual return predictability with machine learning — but whether it anticipates which commodities will move together. If managed money have information about a common driver across a group of commodities — an energy supply shock, a drought, an industrial slowdown — they should position coordinately on that group before prices reflect it. The co-movement structure of their positioning should therefore precede the co-movement structure of realized returns.

---

## Objectives

*Do clusters formed from managed money COT positioning at $t-h$ predict clusters formed from realized returns at $t$, and at what horizon?*

- **Signal existence** — is the mean ARI between COT and return clusters significantly above zero after correcting for serial autocorrelation?
- **Optimal horizon** — at which lag $h$ does the signal peak?
- **Signal heterogeneity** — does the signal vary across market regimes (crisis vs calm) and COT trader categories?

---

## Data and Method

**Universe** — 16 commodity futures, continuous front-month series, 2006–2024. Source: Yahoo Finance. Weekly log returns are computed from Tuesday closes, aligned with the COT reference date. Platinum and palladium were excluded due to extensive missing data before 2010.

| Sector | Tickers |
|---|---|
| Energy | CL (WTI Crude), NG (Natural Gas), RB (RBOB Gasoline) |
| Metals | GC (Gold), SI (Silver), HG (Copper) |
| Grains | ZC (Corn), ZW (Wheat), ZS (Soybean) |
| Softs | KC (Coffee), SB (Sugar), CC (Cocoa), CT (Cotton) |
| Livestock | LE (Live Cattle), HE (Lean Hogs), GF (Feeder Cattle) |

**COT data** — CFTC Disaggregated Report, weekly, 2006–2024, matched by CFTC contract code. Two features are extracted from the Managed Money category per commodity per week, then z-scored on a 52-week rolling window:

- `MM_ratio` : (MM Long − MM Short) / (MM Long + MM Short)
- `ΔMM_ratio` : MM_ratio(t) − MM_ratio(t−4)

**Two clustering objects** are built each week with Ward linkage and cut at $k = 5$ clusters (motivated by the five sectors and validated by the silhouette score):

- *COT positioning clusters* — Euclidean distance on the z-scored COT features (16 commodities × 2 features).
- *Return co-movement clusters* — correlation distance $D_{ij} = \sqrt{2(1-\rho_{ij})}$, with $\rho_{ij}$ the Spearman correlation of the past 16 weekly returns. This is a proper Euclidean distance, consistent with Ward linkage.

**Lead-lag test** — for each week $t$ and lag $h \in \{0, 4, 8, 12\}$ weeks (0–3 months):

```
ARI_h(t) = ARI(C_{t-h}^COT, C_t^ret)
lift_h   = mean(ARI_h) - mean(ARI_0)
```

Significance is assessed with Newey-West HAC t-statistics to account for serial autocorrelation from overlapping windows.

**Sensitivity analyses:**

- Five COT trader category specifications: M1_MM (baseline), M2_PM (Producer/Merchant), M3_SD (Swap Dealers), M4_OR (Other Reportables), M5_combined (all four)
- Subperiod decomposition across 4 crisis periods (GFC 2008–09, Oil crash 2014–15, COVID 2020, Gas shock 2021–22) and 3 calm periods (2016–17, 2018–19, 2023), interpreted descriptively

---

## Results

All outputs are saved in `Results/`. The ARI sample covers 891 weeks, from late 2007 to end-2024.

| Output | Content |
|---|---|
| `summary_table` | Mean ARI (M1_MM): 0.050 (h=0), 0.055 (1m), 0.062 (2m), 0.056 (3m). Newey-West t-stats 7.5–8.6. Peak lag: 2 months. |
| `histogram_ari_h=2` | Weekly ARI at 2m: mean 0.062, median 0.044, std 0.118, 67% of weeks positive, right tail up to 0.62. |
| `dendrogram_h=*` | COT vs return dendrograms for the highest-ARI week at 2m (2024-06-18, ARI = 0.618) — an extreme week, not a typical one. |
| `cat_results_table`, `bar_ari_category`, `heatmap_tstat_category` | Managed Money is the most informative category (~0.06); PM, SD and OR stay around 0.02. Only M1_MM shows a positive lift at 2m (+0.0115, t = 1.97). |
| `sub_by_type_table`, `bar_ari_subperiod`, `heatmap_tstat_subperiod` | Calm periods average 0.078 vs 0.027 in crises (h=0). COVID 2020 shows no signal. |

**Summary:** Managed money positioning clusters significantly track future return co-movement (mean ARI ≈ 0.06, t ≈ 8.5), peaking at two months; the signal is weak, episodic, stronger in calm periods, lift marginal.

---

## Limitations

- **Data:** Yahoo Finance front-month series are not roll-adjusted, so contract rolls introduce artificial price jumps. Back-adjusted series are planned for v2.
- **Lead-lag overlap:** return clusters use the past 16 weeks, so for $h < 16$ weeks part of the return window precedes the COT date. Short lags partly capture a contemporaneous relationship.
- **Choice of $k$:** the ARI depends on $k = 5$. The Mantel test on continuous distance matrices does not require a choice of $k$ and is planned for v2.
- **Subperiods:** too short for formal inference and defined ex post; results are descriptive.
- **No trading implementation:** ARI measures agreement between cluster structures, not economic performance. No backtest is performed.

---

## References

Wang, S., & Zhang, T. (2024). Predictability of Commodity Futures Returns with Machine Learning Models. *Journal of Futures Markets*, 44(2), 302–322. https://doi.org/10.1002/fut.22471
