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
