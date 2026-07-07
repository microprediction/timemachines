# Benchmarking streaming univariate anomaly detection — SOTA survey (2026-07)

Verified against primary sources 2026-07-07 (GitHub/PyPI APIs, official pages,
papers). Context: benchmarking `skaters.mahalanobis` — online one-pass, no
training phase, calibrated per-observation p-value from the Mahalanobis
distance of multi-horizon parade residuals (issue #88).

## Datasets

### UCR Anomaly Archive (UCR-AA) — the credibility test
- 250 univariate series (~6.7k–900k points), exactly one anomaly per series.
- Download: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
  (verified live; password inside the bundled briefing doc, Keogh convention).
- Filename encodes `trainLen_anomStart_anomEnd`; prefix up to `trainLen` is
  guaranteed anomaly-free → use as warm-up only.
- **Scoring: single argmax location.** Report one index/series; hit iff within
  the anomaly range extended by tolerance max(100, anomaly length); accuracy =
  fraction of 250 hit. No thresholds anywhere — suits an argmax-of-d2 method.
- No v2/successor as of 2026-07. Critique lineage: Wu & Keogh, IEEE TKDE 2023
  (arXiv:2009.13807).

### TSB-AD (NeurIPS 2024 D&B) — the leaderboard standard
- https://github.com/TheDatumOrg/TSB-AD (`pip install TSB-AD`, Apache-2.0,
  actively maintained). Leaderboard: https://thedatumorg.github.io/TSB-AD/
- **TSB-AD-U**: 350 univariate eval series + 48-series tuning split (tune
  k/params there, never on eval). Data: https://www.thedatum.org/datasets/TSB-AD-U.zip
- Primary metric **VUS-PR**. Verified univariate top (2026-07): Sub-PCA 0.42,
  KShapeAD 0.40, POLY 0.39 — simple statistical methods beat deep/foundation
  models. Beatable territory.

### Others
- **TimeEval/GutenTAG** (VLDB 2022, MIT, maintained): 71 dockerized detectors;
  GutenTAG synthesizes 10 anomaly types — use for the controlled
  point-outlier-vs-changepoint disambiguation experiment.
- **TCPD/TCPDBench** (Turing Institute): where changepoint capability is
  actually scored (multi-annotator Covering/F1) — optional but high-leverage
  for the run-length half of the method.
- **EasyTSAD/TimeSeriesBench** (ISSRE 2024): only 2024-26 benchmark with
  event-based delay-constrained *online* metrics. Canonical repo:
  https://github.com/dawnvince/EasyTSAD
- **NAB**: window labels, streaming-native design but discredited label
  quality; MIT-relicensed 2024-12, frozen. **Yahoo S5**: host unresolvable
  2026-07, 86% one-liner-solvable — avoid.
- Verified gap: **no major 2024-26 benchmark runs a strictly prequential
  no-split streaming protocol.**

## Metrics

- **Point-adjust F1 is dead** (Kim et al. AAAI 2022: random scores achieve
  SOTA under PA). TSB-AD and TAB both dropped it. Do not headline it.
- **VUS-ROC/VUS-PR** (PVLDB 2022; https://github.com/TheDatumOrg/VUS):
  range-aware threshold-free AUC — de facto leaderboard metric. AUC-family ⇒
  invariant to monotone transforms ⇒ **blind to calibration**.
- **Affiliation P/R** (KDD 2022): event-local, parameter-free; known flaws
  (saturation, weak FP penalization). Co-report, don't rely on.
- **UCR accuracy**: argmax + tolerance, as above.
- Modern reporting bundle: AUC-ROC/PR + VUS-ROC/PR + Affiliation-F1 (+ UCR
  accuracy where applicable).
- **Verified gap = our differentiator: no mainstream metric rewards
  calibration.** Report an additional panel: empirical false-positive rate vs
  nominal alpha on anomaly-free stretches; online FDR (LORD/e-LOND, conformal
  e-values NeurIPS 2023) as the framing literature.

## Streaming baselines (implementation status verified 2026-07)

| Method | Impl | Notes |
|---|---|---|
| SPOT/DSPOT (KDD 2017) | `pip install libspot` 3.0.0 (2026-04, by Siffer) | EVT-calibrated thresholds — closest rival in spirit; documented masking failure under contaminated calibration |
| DAMP (KDD 2022) | `damp.py` in TSB-UAD; no maintained pip | streaming left-discord matrix profile |
| RRCF (ICML 2016) | `pip install rrcf` 0.4.4 (unmaintained but stable) | CoDisp score |
| Half-Space Trees | `river` 0.25.0 (active) | plus river's GaussianScorer / PredictiveAnomalyDetection / QuantileFilter for the trivial baselines |
| Moving z-score | compose in river | community-mandatory trivial baseline |
| MERLIN (batch skyline) | `aeon` 1.5.0 | multi-pass; reference only |

Skip HTM (source-build only, legacy). Aggregators: river (best maintained),
PySAD 0.5.0 (revived 2026-05).

## Recommended plan

1. **UCR-AA, all 250** — argmax of d² (or -log p, monotone-equivalent),
   anomaly-free prefix as warm-up only. Report UCR accuracy.
2. **TSB-AD-U eval split** — VUS-PR/AUC/Affiliation via their package; tune
   only on the 48-series tuning split.
3. **Calibration panel** (the novel part): empirical FPR vs nominal alpha;
   contaminated-warmup sensitivity — verified as an unstudied gap, and the
   winsorized-null masking resistance is already built and tested. Head-to-head
   contamination experiment vs DSPOT.
4. **GutenTAG/TCPD** (optional): point-outlier vs changepoint disambiguation
   via run-length — a capability TSAD benchmarks cannot score.

### Protocol pitfalls
- Strictly causal: never normalize/threshold with whole-file statistics
  (arXiv:2502.05392 flags batch normalization as the main deploy-vs-eval gap).
- Oracle-threshold F1 hands competitors free calibration: also report F1 at
  our fixed p<alpha vs their fixed defaults.
- Map both outlier and changepoint output to the score for range metrics;
  demonstrate the disambiguation separately.
- Pure-Python laplace over UCR-AA's ~25M points is a long run: subset first,
  parallelize per series.

### Flagged as unverifiable
Yahoo Webscope pipeline (host dead); exact UCR briefing-doc wording
(password-protected); TAB license (none detected).
