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
