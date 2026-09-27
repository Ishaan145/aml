"""Hypothesis checks that decide the pipeline design. Run BEFORE building blocking.

    python eda_check.py --data-dir student_resource/dataset  [--sample 30000]

Every section prints a DECISION line telling you which branch of the plan to take.
Needs: pandas, numpy, scikit-learn. ~2-5 min and a few GB RAM on the full data.
"""
import argparse
import csv
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RNG = np.random.default_rng(42)


def read_tsv(path):
    # QUOTE_NONE: an address containing a stray '"' would otherwise swallow following rows.
    # keep_default_na=False: a business literally named "NA" or "None" must stay a string.
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, encoding="utf-8-sig",
                     keep_default_na=False, na_filter=False, on_bad_lines="warn")
    with open(path, encoding="utf-8-sig") as f:
        raw = sum(1 for line in f if line.strip()) - 1
    flag = "" if raw == len(df) else f"   <-- PARSE MISMATCH: {raw} raw lines vs {len(df)} rows"
    print(f"  {path.split('/')[-1]:<24} rows={len(df):>9,}{flag}")
    return df


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def digit_tokens(s):
    return set(re.findall(r"\d+", s))


def postal(s):
    # country-agnostic guess: 5-6 digit token (US ZIP, India PIN, FR code postal); 'NNN NNN' PIN form too
    s = re.sub(r"\b(\d{3})\s(\d{3})\b", r"\1\2", s)
    toks = re.findall(r"\b\d{5,6}\b", s)
    return toks[-1] if toks else None


def pct(x, n):
    return f"{100 * x / max(n, 1):6.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--sample", type=int, default=30000)
    a = ap.parse_args()
    d = a.data_dir.rstrip("/")

    print("\n== 0. Files ==")
    tr = {k: read_tsv(f"{d}/train/train_source{k}.tsv") for k in (1, 2, 3)}
    te = {k: read_tsv(f"{d}/test/test_source{k}.tsv") for k in (1, 2, 3)}
    gt_df = read_tsv(f"{d}/train/train_ground_truth.tsv")

    print("\n== 1. Sizes by country (watch the test France share and pool density) ==")
    for name, grp in (("train", tr), ("test", te)):
        for k, df in grp.items():
            vc = df["country"].value_counts()
            print(f"  {name} S{k}: " + ", ".join(f"{c!r}={n:,}" for c, n in vc.items()))
        dens = (len(grp[2]) + len(grp[3])) / len(grp[1])
        print(f"  {name} pool density (|S2|+|S3|)/|S1| = {dens:.2f}")
    labels = set().union(*[set(df["country"]) for df in list(tr.values()) + list(te.values())])
    print(f"  distinct country labels: {sorted(labels)}")
    print("  DECISION: if labels have variants (US/USA/United States) normalize them before any blocking.")

    for name, grp in (("train", tr), ("test", te)):
        for k, df in grp.items():
            en = (df["business_name"].str.strip() == "").mean()
            ea = (df["business_address"].str.strip() == "").mean()
            dup = df["entity_id"].duplicated().sum()
            print(f"  {name} S{k}: empty name {en:.2%}, empty address {ea:.2%}, dup ids {dup}")

    print("\n== 2. Ground truth shape ==")
    gt = {r.source1_entity_id: [x for x in r.matched_entity_ids.split(",") if x]
          for r in gt_df.itertuples(index=False)}
    s1_ids = set(tr[1]["entity_id"])
    print(f"  S1 in GT: {len(gt):,} / S1 in file: {len(s1_ids):,}; GT S1 not in file: {len(set(gt) - s1_ids)}")
    sizes = Counter(len(v) for v in gt.values())
    n = len(gt)
    print(f"  singleton rate: {pct(sizes[0], n)}")
    for k in sorted(sizes)[:12]:
        print(f"    {k:>3} matches: {sizes[k]:>8,} ({pct(sizes[k], n)})")
    s2n = sum(x.startswith("S2-") for v in gt.values() for x in v)
    s3n = sum(x.startswith("S3-") for v in gt.values() for x in v)
    print(f"  matched ids from S2: {s2n:,}   from S3: {s3n:,}")
    same_src_multi = sum(1 for v in gt.values()
                         if max(Counter(x[:2] for x in v).values(), default=0) >= 2)
    print(f"  S1s with >=2 matches from the SAME source: {pct(same_src_multi, n)}")
    print("  DECISION: singleton rate sets the value of the 'predict empty' decision; "
          "same-source multiplicity says whether to allow >1 match per source.")

    print("\n== 3. H1 exclusivity: does any S2/S3 record belong to >1 S1? ==")
    owner = Counter(x for v in gt.values() for x in v)
    multi = sum(1 for c in owner.values() if c > 1)
    print(f"  S2/S3 ids in >1 GT list: {multi:,} of {len(owner):,}")
    print("  DECISION: if 0 -> enforce 'each S2/S3 record keeps only its best S1' (reverse argmax). "
          "If >0 -> do NOT enforce it.")

    print("\n== 4. Distractor rate: S2/S3 records that match nothing ==")
    for k in (2, 3):
        unmatched = (~tr[k]["entity_id"].isin(owner.keys())).mean()
        print(f"  train S{k} unmatched: {unmatched:.2%}")

    print("\n== 5. H2 country: do true pairs ever cross country labels? ==")
    ctry = {}
    for df in tr.values():
        ctry.update(zip(df["entity_id"], df["country"]))
    pairs = [(s1, x) for s1, v in gt.items() for x in v]
    cross = sum(1 for s1, x in pairs if ctry.get(s1) != ctry.get(x))
    print(f"  cross-country true pairs: {cross:,} / {len(pairs):,}")
    print("  DECISION: if ~0 -> hard-partition blocking by country (cheap, shrinks candidates).")

    print(f"\n== 6. Signal in true pairs (sample {a.sample:,}) vs random same-country pairs ==")
    rec = {}
    for df in tr.values():
        rec.update(zip(df["entity_id"], zip(df["business_name"], df["business_address"], df["country"])))
    idx = RNG.choice(len(pairs), size=min(a.sample, len(pairs)), replace=False)
    tp = [pairs[i] for i in idx]
    pool = pd.concat([tr[2], tr[3]])
    by_c = {c: g["entity_id"].to_numpy() for c, g in pool.groupby("country")}
    rp = []
    for s1, _ in tp:
        c = rec[s1][2]
        if c in by_c:
            rp.append((s1, by_c[c][RNG.integers(len(by_c[c]))]))

    def stats(pp, tag):
        nm_a = [norm(rec[p][0]) for p, _ in pp]
        nm_b = [norm(rec[q][0]) for _, q in pp]
        ad_a = [norm(rec[p][1]) for p, _ in pp]
        ad_b = [norm(rec[q][1]) for _, q in pp]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
        vec.fit(nm_a + nm_b + ad_a + ad_b)
        cos_n = np.asarray(vec.transform(nm_a).multiply(vec.transform(nm_b)).sum(1)).ravel()
        cos_a = np.asarray(vec.transform(ad_a).multiply(vec.transform(ad_b)).sum(1)).ravel()
        exact = np.mean([x == y for x, y in zip(nm_a, nm_b)])
        tok0 = np.mean([not (set(x.split()) & set(y.split())) for x, y in zip(nm_a, nm_b)])
        pz = [(postal(rec[p][1]), postal(rec[q][1])) for p, q in pp]
        both = [(x, y) for x, y in pz if x and y]
        dig = [(digit_tokens(rec[p][1]), digit_tokens(rec[q][1])) for p, q in pp]
        dig_both = [(x, y) for x, y in dig if x and y]
        print(f"  [{tag}] n={len(pp):,}")
        print(f"    name 3-gram cosine  p5/p25/p50: {np.percentile(cos_n, [5, 25, 50]).round(3)}")
        print(f"    addr 3-gram cosine  p5/p25/p50: {np.percentile(cos_a, [5, 25, 50]).round(3)}")
        print(f"    normalized name exactly equal: {exact:.2%}   zero shared name tokens: {tok0:.2%}")
        print(f"    postal code present on both: {len(both) / len(pp):.2%}; "
              f"equal when both present: {np.mean([x == y for x, y in both]) if both else float('nan'):.2%}")
        print(f"    any shared digit token (both have digits): "
              f"{np.mean([bool(x & y) for x, y in dig_both]) if dig_both else float('nan'):.2%}")
        return cos_n, cos_a

    tn, ta = stats(tp, "TRUE pairs")
    rn, ra = stats(rp, "RANDOM same-country pairs")
    hard = np.mean((tn < 0.3) & (ta < 0.3))
    dba = np.mean(tn < 0.2)
    print(f"  true pairs with name<0.2 (DBA-like): {dba:.2%}; with BOTH name<0.3 and addr<0.3: {hard:.2%}")
    print("  DECISION: DBA-like share -> you need an address-only retrieval view. "
          "Both-low share ~ your irreducible recall loss for string blocking.")

    print("\n== 7. Eyeball: 3 true pairs per country, 5 test France rows ==")
    for c in sorted({rec[s][2] for s, _ in tp}):
        ex = [(s, q) for s, q in tp if rec[s][2] == c][:3]
        for s, q in ex:
            print(f"  [{c}] {rec[s][0]!r} | {rec[s][1]!r}\n       -> {rec[q][0]!r} | {rec[q][1]!r}")
    for k in (1, 2):
        fr = te[k][~te[k]["country"].isin(set(tr[1]["country"]))].head(5)
        for r in fr.itertuples(index=False):
            print(f"  test S{k} unseen-country row: {r.business_name!r} | {r.business_address!r} | {r.country!r}")


if __name__ == "__main__":
    main()
