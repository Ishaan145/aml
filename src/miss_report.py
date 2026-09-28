"""Why does blocking miss true pairs? Reads the cached work/ files (no re-blocking), ~5 min.

    python -m src.miss_report --work work

Buckets every missed train pair (true match not in the blocked candidates) and prints examples:
  no_shared_token : pair shares no name or address word at all (only fuzzy/char signal could find it)
  purged_only     : every shared word is in more than tok_max_df pool records, so blocking ignored it
  ranked_out      : shares a usable word but lost the top-k race to other records
and, per bucket, how many misses belong to an S1 whose OTHER true matches WERE found, and whether the
missed record looks more like a found sibling than like the S1 (-> sibling expansion would recover it).
"""
import argparse
import os

import numpy as np
import pandas as pd

from .blocking import hash_vectorize

NF = 2 ** 22


def load(work):
    rec = pd.read_parquet(os.path.join(work, "rec_train.parquet"))
    for c in ("id", "country", "name_n", "addr_n", "postal"):
        rec[c] = rec[c].astype(object)
    gt = pd.read_parquet(os.path.join(work, "gt_train.parquet"))
    cand = pd.read_parquet(os.path.join(work, "cand_train.parquet"), columns=["q", "c", "label"])
    q_eval = np.load(os.path.join(work, "qeval_train.npy"))
    return rec, gt, cand, q_eval


def jacc(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a | b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--max-df", type=int, default=20000)
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    rec, gt, cand, q_eval = load(a.work)
    n = len(rec)
    gt = gt[np.isin(gt["q"].values, q_eval)].copy()
    found = set((cand.loc[cand["label"] == 1, "q"].values.astype(np.int64) * n
                 + cand.loc[cand["label"] == 1, "c"].values.astype(np.int64)).tolist())
    key = gt["q"].values.astype(np.int64) * n + gt["c"].values.astype(np.int64)
    gt["found"] = np.fromiter((k in found for k in key), bool, len(key))
    print(f"true pairs for queried S1s: {len(gt):,}   found by blocking: {gt['found'].mean():.4f}")

    miss = gt[~gt["found"]].copy()
    ctry = rec["country"].values
    miss["country"] = ctry[miss["q"].values]
    nm, ad, src = rec["name_n"].values, rec["addr_n"].values, rec["src"].values

    # document frequency of each word inside that country's S2/S3 pool, per field (as blocking sees it)
    miss["min_df_shared"] = np.inf
    miss["n_shared"] = 0
    for c in pd.unique(miss["country"]):
        pool_idx = np.flatnonzero((ctry == c) & (src != 1))
        sel = np.flatnonzero(miss["country"].values == c)
        qs, cs = miss["q"].values[sel], miss["c"].values[sel]
        best_df = np.full(len(sel), np.inf)
        n_sh = np.zeros(len(sel), np.int64)
        for field in (nm, ad):
            P = hash_vectorize(field[pool_idx].tolist(), "word", NF, 4)
            df = np.bincount(P.indices, minlength=NF)
            del P
            A = hash_vectorize(field[qs].tolist(), "word", NF, 4)
            B = hash_vectorize(field[cs].tolist(), "word", NF, 4)
            S = A.multiply(B).tocsr()
            n_sh += np.diff(S.indptr)
            # min df over the shared words of each row (rows with none shared -> inf)
            vals = df[S.indices].astype(np.float64)
            cnt = np.diff(S.indptr)
            m = np.full(len(cnt), np.inf)
            nz = cnt > 0
            if nz.any():
                m[nz] = np.minimum.reduceat(vals, S.indptr[:-1][nz])
            best_df = np.minimum(best_df, m)
        miss.iloc[sel, miss.columns.get_loc("min_df_shared")] = best_df
        miss.iloc[sel, miss.columns.get_loc("n_shared")] = n_sh
    miss["bucket"] = np.where(miss["n_shared"] == 0, "no_shared_token",
                              np.where(miss["min_df_shared"] > a.max_df, "purged_only", "ranked_out"))

    # sibling analysis: other true matches of the same S1 that blocking DID find
    found_pairs = gt[gt["found"]]
    sib = found_pairs.groupby("q")["c"].apply(list).to_dict()
    has_sib, closer = [], []
    for q, c in zip(miss["q"].values, miss["c"].values):
        s = sib.get(q, [])
        has_sib.append(bool(s))
        if s:
            txt = nm[c] + " " + ad[c]
            best = max(jacc(txt, nm[x] + " " + ad[x]) for x in s)
            closer.append(best > jacc(txt, nm[q] + " " + ad[q]))
        else:
            closer.append(False)
    miss["has_found_sibling"] = has_sib
    miss["closer_to_sibling"] = closer

    print(f"missed pairs: {len(miss):,} ({len(miss) / len(gt):.2%} of true pairs)\n")
    tab = miss.groupby(["country", "bucket"]).agg(
        pairs=("q", "size"), share_of_misses=("q", lambda x: len(x) / len(miss)),
        has_found_sibling=("has_found_sibling", "mean"), closer_to_sibling=("closer_to_sibling", "mean"))
    print(tab.round(3).to_string(), "\n")
    s1_all_missed = miss.groupby("q").size()
    n_true = gt.groupby("q").size()
    fully = (s1_all_missed == n_true.reindex(s1_all_missed.index)).sum()
    print(f"S1s losing ALL their true matches in blocking: {fully:,}; S1s losing some: {len(s1_all_missed):,}")
    print(f"misses recoverable by sibling expansion (has found sibling AND closer to it): "
          f"{(miss['has_found_sibling'] & miss['closer_to_sibling']).mean():.2%}")
    rng = np.random.default_rng(0)
    for (c, b), g in miss.groupby(["country", "bucket"]):
        print(f"\n=== {c} / {b}: examples ===")
        for i in rng.choice(len(g), min(a.examples, len(g)), replace=False):
            r = g.iloc[i]
            print(f"  S1: {nm[r.q]!r} | {ad[r.q]!r}\n   ->{nm[r.c]!r} | {ad[r.c]!r}"
                  f"   [shared={r.n_shared}, min_df={r.min_df_shared:.0f}, sibling_found={r.has_found_sibling}]")


if __name__ == "__main__":
    main()
