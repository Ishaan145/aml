# Amazon ML Challenge 2026 Solution

**Team Name:** MathComp

We solve entity resolution as a cascade of increasingly precise stages, each trained only on the provided data

1. **Candidate generation (blocking), per country.** Three sparse retrieval views plus exact keys. The key addition is a **reverse view**, in which every Source 2/3 record retrieves its own best Source 1 entities. It raised pair completeness from 87.9% to 95.8%.
2. **Supervised pruner.** Cuts the candidates to 6.3 per Source 1 entity on the test set.
3. **Matcher.** A LightGBM model that uses string, digit and IDF-overlap features, a fine-tuned multilingual **transformer cross-encoder** score, and that score's rank and gap among competing candidates.
4. **Decision layer.** For each Source 1 entity it chooses the match set that maximises expected F0.5.

**Result:** out-of-fold validation macro F0.5 is **0.9756** (US 0.9818, India 0.9665). The leaderboard (public portal) score is **0.969415** on the hidden test set, which includes the unseen French data. Section 5.2 explains the 0.0062 gap.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA on the full data)

**Data size.**
- Train: S1 2,206,821 · S2 5,034,616 · S3 5,285,603.
- Test: S1 1,732,544 (India 809,986, US 663,106, France 259,452 — France is unseen in training) · S2 4,887,273 · S3 5,082,316.
- The test pool is denser than train: 5.75 S2/S3 records per S1, against 4.68.

**Label structure.**
- Singletons (S1 with no match): 5.58%. Otherwise S1s have on average 3.5 matches, and 76.8% of S1s have 2 or more matches from the *same* source.
- **No S2/S3 record belongs to more than one S1**, so each S2/S3 record is allowed only its best S1.
- **No true pair crosses country labels**, so blocking is partitioned by country.

**Noise found in the data.**
- 14.5% of true pairs share no name word. These are trade names and website-style names (`apexallied.com` vs "Apex Allied Hospitality").
- Postcodes are present on both sides of only 5.7% of true pairs.
- 3.4% of S2/S3 addresses are empty.
- Names and states appear in Indic scripts (Devanagari, Tamil, Bengali, Telugu, Gujarati, Kannada).
- Words are reordered ("STAR LIMITED PRIVATE TECH"), characters are typo'd, and even house numbers are noisy.

**Error analysis that drove the design.** On the first full run, 12.1% of true pairs were missed by blocking. Our `miss_report` tool bucketed them:
- 91% were *ranked out*: the pair shares informative words, but S1 entities with many matches and many look-alike records pushed the true record below the top-k.
- 9% shared only very common words. These were mostly Indic-script names that normalization had mangled.
- 0.04% shared no word at all.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — multi-view blocking with a reverse view, a supervised pruner, a gradient-boosted matcher with transformer cross-encoder features, and a decision-theoretic set decoder.

**Core Innovations:**
1. **Reverse (record-to-entity) retrieval.** Each S2/S3 record queries an index of all S1 entities of its country and keeps its top 5. Seen from the record's side, there are far fewer competitors, and exclusivity holds. This view alone reaches 94.1% recall, against 80.8% for forward word-token retrieval. The union rose from 87.9% to 95.8%.
2. **Script-aware normalization.** Indic scripts are transliterated to Latin with `anyascii` (ISC licence), so "यूनिक इन्वेस्टमेंट प्राइवेट लिमिटेड" becomes "yunik investmemt praivet limited". Token equivalences such as texas↔tx, maharashtra↔mh and maharashtra↔mharastr are *mined* from matched training pairs (161 learned), not hand-written.
3. **Listwise cross-encoder features.** A fine-tuned `intfloat/multilingual-e5-small` cross-encoder (MIT, 118M parameters) scores each pair. The matcher also receives that score's rank and gap among the S1's candidates, and among the S1s competing for the same record. This is the "selective" matching idea of Mistral4SelectEM (Advanced Engineering Informatics, 2025), realised cheaply with a small model.
4. **Expected-F0.5 set decoder.** For each S1, the decoder picks the top-k (k = 0…n) by calibrated probability that maximises expected F0.5, computed exactly with a Poisson-binomial dynamic programme. k = 0 is scored as P(no match), so singletons are decided explicitly. It beat a tuned global threshold on validation.

```
raw TSVs → normalize (anyascii transliteration, mined equivalences, postcode, per-country generic tokens)
 → per country: forward word-token IDF top-20 ∪ reverse record→S1 top-5 ∪ exact keys → reciprocal-rank fusion
 → LightGBM pruner (supervised meta-blocking) → candidate_pairs.tsv
 → features + cross-encoder score and listwise context → LightGBM matcher → isotonic calibration
 → S2/S3 exclusivity → expected-F0.5 decoder → matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

**Blocking keys and views**, all computed per country; country is never used as a model feature:

| View | What it indexes | Recall alone (train probe) |
|---|---|---|
| Forward word-token IDF | name + address + postcode; field weights 1.0 / 0.6 / 1.2; cosine top-20 per S1 | 80.8% |
| Reverse retrieval | every S2/S3 record → its top-5 S1s | 94.1% |
| Exact keys | sorted name core; name + postcode; postcode + name prefix; postcode + house number; house number + first two address words (± name prefix); house number + two rarest address words | 70.0% |
| **Union** | reciprocal rank fusion, capped at 40 per S1; very specific key hits and top-2 reverse hits always kept | **95.8%** |

**Scaling measures.**
- Sparse matrix top-k retrieval (`sparse_dot_topn`).
- Cost-budgeted block purging: tokens in more than 20,000 pool records are dropped, and the cut-off tightens automatically if the estimated postings per query exceed a budget.
- Exact keys are stored as 64-bit hashes.
- Queries are processed in chunks with progress logs.
- Full train and test blocking took about 2.6 hours on a 4-core laptop.

**Supervised pruning.** The last filtering stage is a LightGBM model trained with cross-fitting grouped by S1 entity. Its features are blocking scores, fast string similarities and competition features. Its cut-off keeps 99.8% of blocked true pairs, with at most 20 candidates per S1.

**Test candidate pairs generated:** **10,861,533**.
- 6.27 candidates per S1 entity on average (95th percentile: 11).
- 2,808 S1 entities with zero candidates.
- Reduction ratio against all same-country pairs: **0.99999838**.

**How true matches were not lost.**
- The views fail in different ways: the forward view misses on look-alikes, the reverse view on sparse records, and the keys on typos. Their union covers each view's gaps.
- Every change was checked with the `probe` stage (5% of train S1s against the full pool) and the `miss_report` diagnostic.
- Final pair completeness on validation is **95.6%** after pruning.

---

## 4. Matching Model

**Features used.**

*Name*
- rapidfuzz ratio, token-sort, token-set and partial ratios, and Jaro-Winkler, on the normalized name and on the "name core" (generic tokens removed).
- Token Jaccard, exact and core equality, acronym match, symmetric-difference size.
- IDF-weighted overlap (shared IDF mass, Jaccard, coverage of each side, and the IDF of the rarest word present on only one side), using per-country IDF.

*Address*
- The same string and IDF-overlap measures.
- House number equal, different or missing; numeric-token Jaccard and counts; postcode and 3-digit postcode-prefix agreement.

*Transformer cross-encoder*
- Match logit of the fine-tuned `multilingual-e5-small` on raw "name | address" pairs.
- Its rank, gap and second-best value among the S1's candidates.
- Its rank and gap among the S1s competing for the same record, plus a mutual-best flag.

*Other*
- Blocking scores and ranks, key bits and block size, and reverse-view rank and gap.
- Pruner probability and its within-S1 context.
- Name frequency among S1 and pool records, which flags chains and franchises.
- Source (S2 or S3).

**Cross-encoder training.**
- 1.5M pairs (35.6% positive) mined with the forward view and keys from 200,000 train S1 entities that the matcher does not use.
- 1 epoch, AdamW, learning rate 5e-5, max length 128, fp16, on 2×T4 GPUs in 48 minutes. Loss fell from 0.46 to 0.024.
- It then scored all 17M matcher pairs (train and test) with one process per GPU.

**Model type.**
- LightGBM binary classifier: 127 leaves, learning rate 0.1, early stopping.
- 2-fold cross-fitting grouped by S1 entity, then refit on all training rows.
- The matcher trains on a looser candidate set than inference uses, to see more hard negatives.
- Feature gain: cross-encoder 71%, within-S1 cross-encoder gap 15%, pruner probability 8%.

**Threshold selection method.**
- Isotonic calibration on out-of-fold predictions.
- The expected-F0.5 set decoder and a global threshold, each with and without exclusivity, are scored by exact macro F0.5 over *all* validation S1 entities, singletons included. The best is chosen.
- Chosen: expected-F0.5 decoder with exclusivity.

---

## 5. Results & Error Analysis

### 5.1 Scores

Macro F0.5 on out-of-fold validation: ≈40% of train S1 entities (about 0.88M entities with 3,054,665 true pairs), all with US/India labels.

| System | Validation macro F0.5 | US | India |
|---|---|---|---|
| First full pipeline (forward view + char n-grams + keys) | 0.937 | 0.951 | 0.916 |
| + reverse view, Indic transliteration, `.com` splitting | 0.9708 | 0.9777 | 0.9604 |
| + transformer cross-encoder features (**submitted**) | **0.9756** | **0.9818** | **0.9665** |

| Submitted system, by component (validation) | Value |
|---|---|
| Singleton share / singleton accuracy | 5.59% / 0.9862 (0.9665 without the cross-encoder) |
| Non-singleton F0.5 | 0.9750 |
| Pair completeness after blocking / after pruning | 95.77% / 95.58% |
| Chosen decoder | expected-F0.5, with S2/S3 exclusivity |

**Leaderboard (portal) score of the submitted file: 0.969415.**

### 5.2 Validation vs leaderboard: why 0.9756 became 0.9694

The portal score is 0.0062 lower than our validation score. The test set has no labels, so the split below
is partly measured and partly estimated. Each line says which.

| Cause | Effect on the gap | Status |
|---|---|---|
| **Different country mix.** Test S1 is 46.8% India, 38.3% US and 15.0% France. Validation is 40.0% India, 60.0% US, 0% France. India scores lower than US (0.966 vs 0.982), so more India weight lowers the average. | Re-weighting our own per-country validation scores to the test's US/India mix gives 0.9733, i.e. **−0.0023** | **Measured** from our validation numbers |
| **France is unseen.** 259,452 test S1 entities (15.0%) come from a country with no training labels. Nothing in validation measures them. | If US and India generalise exactly as validated, the portal score implies France F0.5 ≈ **0.947**, i.e. roughly **−0.0039** | **Estimate**: it assumes zero drop on US/India, so the real France score may be higher |
| **Denser test pool.** 5.75 S2/S3 records per S1 in test vs 4.68 in train, and 6.27 vs 5.13 candidates per S1 after pruning. More look-alike candidates give more chances for a false merge. | Not separable from the France term | **Plausible**, not measured |
| **Optimism in validation.** (a) The decoder and its threshold were chosen on the same out-of-fold predictions they are scored on. (b) About 3% of the matcher's training entities overlap the cross-encoder's training entities (we used a 30% exclusion sample for the cross-encoder and a 40% sample for the matcher). | Small; it inflates validation only | **Known**, size not measured |
| **Public leaderboard is a subset of the test set.** | Sampling noise only. With hundreds of thousands of entities this is far below 0.006. | Known |

**Takeaway:** about a third of the gap (0.0023) is explained by country mix alone, and our own numbers
measure that part. Most of the rest is consistent with lower accuracy on the unseen French data. Possible
improvements: a France-specific error analysis (for example "R." vs "Rue", "Bd" vs "Boulevard",
SARL/SAS/EURL suffix handling), and validation that reproduces the test's country mix and pool density.

### 5.3 Error analysis

- **Remaining recall loss is mostly in blocking.** 4.2% of true train pairs never become candidates.
  Even with a perfect matcher, this alone caps F0.5 at about 0.99.
- **Common false positives (wrong merges):**
  - branches of the same chain on the same street;
  - different businesses at one address with generic names ("Health Coalition", "Vision Group").
- **Common false negatives (missed matches):**
  - records with an empty address plus a trade or website name (`highcom` vs "High Group");
  - Indic-script names whose transliteration diverges strongly (`piraivet limitet`);
  - S1 entities with 6 or more matches, where look-alike records crowd the candidate list.
- **Blocking misses on the first full run** (370,571 pairs, 12.1%), from our `miss_report` tool:
  - 91% were *ranked out*: shared informative words, but outranked by look-alikes (fixed by the reverse view);
  - 9% shared only very common words, mostly Indic-script names (fixed by transliteration);
  - 0.04% shared no word at all.

## 6. Conclusion

Diagnosing *why* true pairs were missed, rather than tuning blindly, drove every gain: reverse retrieval and script-aware normalization lifted completeness from 87.9% to 95.8%. A small multilingual cross-encoder, used as listwise features inside a gradient-boosted matcher, added a further 0.005 on validation. On the hidden test set the system scores 0.9694. The 0.006 drop is largely explained by the test's heavier India share and its unseen French data, which points to France-specific normalization as the next improvement. Everything runs on the provided data alone, with permissively licensed models of at most 118M parameters.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`
- `src/pipeline.py` is the entry point, with stages `prep`, `probe`, `block`, `prune` and `match`.

| Module | What it does |
|---|---|
| `io_utils` | robust TSV reading |
| `normalize` | normalization, transliteration, mined equivalences |
| `blocking` | forward, reverse and key views, fusion |
| `features` | pair and context features |
| `models` | cross-fitted LightGBM |
| `decode` | calibration, exclusivity, expected-F0.5 decoder |
| `metric` | exact macro F0.5 |
| `ce_model` | cross-encoder: mine, train, score |
| `export_pairs` | exports candidate pairs for the GPU machine |
| `eda_check`, `miss_report`, `diagnose` | analysis tools |

**Reproduction** (see `README.md` for exact commands):
```bash
python -m src.pipeline --data-dir $DATA --work work --stages prep,block,prune,match --train-frac 0.4 --set n_jobs=4
python -m src.pipeline --data-dir $DATA --work work_gpu --stages prep                                   # GPU machine
python -m src.ce_model mine  --data-dir $DATA --work work_gpu --matcher-train-frac 0.3 --n-s1 200000 --topk 8
python -m src.ce_model train --data-dir $DATA --pairs work_gpu/ce_mine_pairs.parquet --model-out ce_model
python -m src.export_pairs --work work --out ce_export
python -m src.ce_model score --data-dir $DATA --split test  --pairs ce_export/pairs_test.parquet  --align ce_export/align.json --model-dir ce_model --out work/ce_test.parquet
python -m src.ce_model score --data-dir $DATA --split train --pairs ce_export/pairs_train.parquet --align ce_export/align.json --model-dir ce_model --out work/ce_train.parquet
python -m src.pipeline --data-dir $DATA --work work --stages match --set n_jobs=4   # -> ../../output/matching_results.tsv
```
The Kaggle notebook used for the GPU steps is included as `notebooks/amazon_ml_2026_ce_kaggle.ipynb`.

### B. Additional Results

- **Decision table (with cross-encoder):**

  | Decoder | Exclusivity | Macro F0.5 |
  |---|---|---|
  | expected-F0.5 | on | 0.97563 |
  | expected-F0.5 | off | 0.97561 |
  | threshold 0.69 | on | 0.97547 |
  | threshold 0.69 | off | 0.97546 |

- **Blocking progression:** 87.9% pair completeness (forward view + keys + char n-grams) → 95.8% (forward view + reverse view + keys, with transliteration).
- **Pruner:** 765 rounds. Most important features: record-side RRF rank (0.44), record-side RRF gap (0.22), address token-set ratio (0.14).
