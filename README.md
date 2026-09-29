# TEAM MATHCOMP AMAZOM ML SOLUTION

Multi-view blocking (forward + **reverse** retrieval + exact keys) → supervised pruner → LightGBM matcher
with **transformer cross-encoder** features → calibrated, per-entity expected-F0.5 decision.
Regenerates `output/candidate_pairs.tsv` and `output/matching_results.tsv` from the provided TSVs.

**Data and models.** Only the provided training/test files are used: no external data, no lookup APIs and
no geocoding. The only pretrained weights are `intfloat/multilingual-e5-small` (MIT licence, 118M
parameters), downloaded once from the Hugging Face Hub and fine-tuned on the training data. LightGBM (MIT)
and anyascii (ISC) are the other modelling libraries.

## Layout

```
src/
  pipeline.py      entry point; stages prep → probe (optional) → block → prune → match
  config.py        every tunable knob (override with --set key=value); defaults = submitted run
  io_utils.py      robust TSV reading (no quote parsing, "NA" stays a string), TSV writing
  normalize.py     normalization, anyascii transliteration of Indic scripts, mined token equivalences,
                   postcode extraction, per-country generic tokens
  blocking.py      per-country views: forward word-token IDF top-k, reverse record→S1 top-k, exact keys;
                   reciprocal-rank fusion; cost-budgeted block purging
  features.py      string, digit, IDF-overlap, name-frequency and competition (listwise) features
  models.py        LightGBM with cross-fitting grouped by S1 entity
  decode.py        isotonic calibration, S2/S3 exclusivity, threshold vs expected-F0.5 set decoder
  metric.py        exact macro F0.5 (challenge definition, singletons included)
  ce_model.py      cross-encoder: mine hard negatives, fine-tune, score pairs (GPU)
  export_pairs.py  exports the matcher's candidate pairs for scoring on the GPU machine
  eda_check.py     data checks;  miss_report.py / diagnose.py  blocking-miss analysis
notebooks/
  amazon_ml_2026_ce_kaggle.ipynb   the Kaggle (2×T4) notebook used for the cross-encoder steps
```

## Hardware used for the submitted run

- **CPU stages:** a Windows laptop with 4 cores and ≥ 30 GB RAM recommended, Python 3.13. Wall time:
  prep 10 min, block 2.6 h, prune 43 min, match 17 min.
- **GPU stages:** Kaggle notebook with 2× Tesla T4 (torch 2.10, transformers 5.0). Wall time: prep 11 min,
  mine 6 min, train 48 min, scoring of ~17M pairs about 1.5 h.

## Install

```bash
pip install -r requirements.txt
```
