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

## Reproduce the submitted result (run from this folder)

```bash
DATA=/path/to/student_resource/dataset     # folder containing train/ and test/

# 1. CPU: normalize, block, prune, baseline match
#    -> ../../output/candidate_pairs.tsv (final candidate set) + baseline matching_results.tsv
python -m src.pipeline --data-dir $DATA --work work --stages prep,block,prune,match --train-frac 0.4 --set n_jobs=4

# 2. GPU: train the cross-encoder on train S1s outside a 30% sample.
#    Use a separate work folder, or the same machine as step 1.
python -m src.pipeline --data-dir $DATA --work work_gpu --stages prep
python -m src.ce_model mine  --data-dir $DATA --work work_gpu --matcher-train-frac 0.3 --n-s1 200000 --topk 8
python -m src.ce_model train --data-dir $DATA --pairs work_gpu/ce_mine_pairs.parquet \
       --model intfloat/multilingual-e5-small --model-out ce_model --max-pairs 1500000 --epochs 1 \
       --batch 128 --max-len 128 --lr 5e-5

# 3. Export the matcher's candidate pairs (CPU) and score them with the cross-encoder (GPU).
#    align.json makes the GPU machine verify that it reads the records in the same order.
python -m src.export_pairs --work work --out ce_export
python -m src.ce_model score --data-dir $DATA --split test  --pairs ce_export/pairs_test.parquet \
       --align ce_export/align.json --model-dir ce_model --out work/ce_test.parquet  --batch 512
python -m src.ce_model score --data-dir $DATA --split train --pairs ce_export/pairs_train.parquet \
       --align ce_export/align.json --model-dir ce_model --out work/ce_train.parquet --batch 512
#    (with 2 GPUs: run one process per GPU with CUDA_VISIBLE_DEVICES=i --part i --parts 2,
#     then the same command with --parts 1 to merge; the notebook does this)

# 4. CPU: final match with cross-encoder features -> ../../output/matching_results.tsv (submitted file)
#    Reports: work/match_report_ce.json (validation macro F0.5 0.9756); baseline: work/match_report.json (0.9708)
python -m src.pipeline --data-dir $DATA --work work --stages match --set n_jobs=4

# 5. Validate (from student_resource/)
python utils/validate_submission.py --matching <path>/output/matching_results.tsv \
       --candidate <path>/output/candidate_pairs.tsv --test-dir dataset/test
```
