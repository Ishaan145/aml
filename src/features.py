"""Pair features for the pruner (cheap) and the matcher (full), plus set-level context features.

Context features are what separate chains/franchises: they tell the model how this candidate
ranks among all candidates of the S1 entity, and how this S1 entity ranks among all S1 entities
competing for the same S2/S3 record.
"""
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

_DIG = re.compile(r"\d+")

BLOCK_FEATS = ["score_tok", "rank_tok", "score_chr", "rank_chr", "score_rev", "rank_rev", "rev_gap",
               "key_bits", "key_block", "n_views", "rrf"]


def _pairs(rec, df, col):
    v = rec[col].values
    return v[df["q"].values], v[df["c"].values]


def _fz(a, b, scorer, n_jobs, scale=100.0):
    return (cpdist(a, b, scorer=scorer, workers=n_jobs) / scale).astype(np.float32)


def add_context(df, score, prefix, reverse=True):
    """Rank/gap of `score` within each S1 (q) and, if reverse, within each candidate record (c)."""
    g = df.groupby("q")[score]
    df[f"{prefix}_rank_q"] = g.rank(ascending=False, method="min").astype(np.float32)
    df[f"{prefix}_gap_q"] = (g.transform("max") - df[score]).astype(np.float32)
    df[f"{prefix}_second_q"] = _second_best(df, "q", score)
    if not reverse:
        return df
    h = df.groupby("c")[score]
    df[f"{prefix}_rank_c"] = h.rank(ascending=False, method="min").astype(np.float32)
    df[f"{prefix}_gap_c"] = (h.transform("max") - df[score]).astype(np.float32)
    df[f"{prefix}_mutual"] = ((df[f"{prefix}_rank_q"] == 1) & (df[f"{prefix}_rank_c"] == 1)).astype(np.int8)
    return df


def _second_best(df, key, score):
    """Second-largest score per group, vectorized (fast path for big frames)."""
    tmp = df[[key, score]].sort_values([key, score], ascending=[True, False])
    pos = tmp.groupby(key).cumcount().values
    second = tmp[pos == 1].set_index(key)[score]
    return df[key].map(second).fillna(0.0).astype(np.float32).values


def cheap_features(df, rec, n_jobs):
    a, b = _pairs(rec, df, "name_n")
    df["nm_ratio"] = _fz(a, b, fuzz.ratio, n_jobs)
    df["nm_tset"] = _fz(a, b, fuzz.token_set_ratio, n_jobs)
    a, b = _pairs(rec, df, "addr_n")
    df["ad_tset"] = _fz(a, b, fuzz.token_set_ratio, n_jobs)
    pq, pc = _pairs(rec, df, "postal")
    df["postal_cmp"] = np.where((pq == "") | (pc == ""), 0, np.where(pq == pc, 1, -1)).astype(np.int8)
    df["src_c"] = rec["src"].values[df["c"].values].astype(np.int8)
    return df


def block_context(df):
    """Competition features from blocking scores. Must run on the FULL query set (every S1),
    before any training subsample, so train and test see the same competition."""
    df["n_cand_q"] = df.groupby("q")["c"].transform("size").astype(np.int16)
    df["n_q_c"] = df.groupby("c")["q"].transform("size").astype(np.int16)
    return add_context(df, "rrf", "rrf")


CHEAP_FEATS = BLOCK_FEATS + ["nm_ratio", "nm_tset", "ad_tset", "postal_cmp", "src_c", "n_cand_q", "n_q_c",
                             "rrf_rank_q", "rrf_gap_q", "rrf_second_q", "rrf_rank_c", "rrf_gap_c", "rrf_mutual"]


def _acronym(x):
    return "".join(t[0] for t in x.split() if t)


def _set_chunk(args):
    nq, nc, cq, cc, aq, ac, pq, pc = args
    out = np.zeros((len(nq), 11), dtype=np.float32)
    for i in range(len(nq)):
        s1, s2 = set(nq[i].split()), set(nc[i].split())
        u = len(s1 | s2)
        out[i, 0] = len(s1 & s2) / u if u else 0.0
        out[i, 1] = nq[i] == nc[i]
        out[i, 2] = cq[i] == cc[i]
        a1, a2 = _acronym(cq[i]), _acronym(cc[i])
        out[i, 3] = (len(a1) >= 2 and a1 in s2) or (len(a2) >= 2 and a2 in s1)
        w1, w2 = set(aq[i].split()), set(ac[i].split())
        u = len(w1 | w2)
        out[i, 4] = len(w1 & w2) / u if u else 0.0
        d1 = [d for d in _DIG.findall(aq[i]) if d != pq[i]]
        d2 = [d for d in _DIG.findall(ac[i]) if d != pc[i]]
        sd1, sd2 = set(d1), set(d2)
        out[i, 5] = len(sd1)
        out[i, 6] = len(sd2)
        ud = len(sd1 | sd2)
        out[i, 7] = len(sd1 & sd2) / ud if ud else -1.0
        out[i, 8] = 0 if not (d1 and d2) else (1 if d1[0] == d2[0] else -1)      # house number
        out[i, 9] = 0 if not (pq[i] and pc[i]) else (1 if pq[i][:3] == pc[i][:3] else -1)
        out[i, 10] = len(s1 ^ s2)
    return out


SET_COLS = ["nm_jacc", "nm_equal", "core_equal", "acronym", "ad_jacc", "n_dig_q", "n_dig_c",
            "dig_jacc", "house_cmp", "postal3_cmp", "nm_symdiff"]


def full_features(df, rec, freq, n_jobs, chunk=200_000):
    a, b = _pairs(rec, df, "name_n")
    df["nm_tsort"] = _fz(a, b, fuzz.token_sort_ratio, n_jobs)
    df["nm_partial"] = _fz(a, b, fuzz.partial_ratio, n_jobs)
    df["nm_jw"] = _fz(a, b, JaroWinkler.normalized_similarity, n_jobs, 1.0)
    df["nm_len_q"] = np.fromiter((len(x) for x in a), np.float32, len(a))
    df["nm_len_c"] = np.fromiter((len(x) for x in b), np.float32, len(b))
    a, b = _pairs(rec, df, "name_core")
    df["core_ratio"] = _fz(a, b, fuzz.ratio, n_jobs)
    df["core_tset"] = _fz(a, b, fuzz.token_set_ratio, n_jobs)
    df["core_jw"] = _fz(a, b, JaroWinkler.normalized_similarity, n_jobs, 1.0)
    a, b = _pairs(rec, df, "addr_n")
    df["ad_ratio"] = _fz(a, b, fuzz.ratio, n_jobs)
    df["ad_tsort"] = _fz(a, b, fuzz.token_sort_ratio, n_jobs)
    df["ad_partial"] = _fz(a, b, fuzz.partial_ratio, n_jobs)
    df["ad_len_q"] = np.fromiter((len(x) for x in a), np.float32, len(a))
    df["ad_len_c"] = np.fromiter((len(x) for x in b), np.float32, len(b))

    cols = [_pairs(rec, df, c) for c in ("name_n", "name_core", "addr_n", "postal")]
    flat = [cols[0][0], cols[0][1], cols[1][0], cols[1][1], cols[2][0], cols[2][1], cols[3][0], cols[3][1]]
    n = len(df)
    jobs = [tuple(x[i:i + chunk] for x in flat) for i in range(0, n, chunk)]
    if n_jobs > 1 and len(jobs) > 1:
        with Pool(n_jobs) as p:
            res = p.map(_set_chunk, jobs)
    else:
        res = [_set_chunk(j) for j in jobs]
    m = np.vstack(res) if res else np.zeros((0, len(SET_COLS)), np.float32)
    for i, c in enumerate(SET_COLS):
        df[c] = m[:, i]

    # how common is this name? (chains/franchises -> address must decide)
    key_q = rec["country"].values[df["q"].values] + "|" + rec["name_core"].values[df["q"].values]
    key_c = rec["country"].values[df["c"].values] + "|" + rec["name_core"].values[df["c"].values]
    df["freq_s1_name"] = pd.Series(key_q).map(freq["s1"]).fillna(0).values.astype(np.float32)
    df["freq_pool_name"] = pd.Series(key_c).map(freq["pool"]).fillna(0).values.astype(np.float32)
    for f in ("name_n", "addr_n"):
        X = idf_matrix(rec, f, n_jobs)
        for k, v in idf_pair_features(X, df["q"].values, df["c"].values).items():
            df[f"{f[:2]}_{k}"] = v
        del X
    return df


def idf_matrix(rec, col, n_jobs, n_features=2 ** 22):
    """Binary token matrix whose values are per-country IDF weights (country-adaptive: frequent
    words such as a business type or 'boulevard' get low weight in whatever country they occur)."""
    from .blocking import hash_vectorize
    X = hash_vectorize(rec[col].tolist(), "word", n_features, n_jobs)
    codes = pd.factorize(rec["country"].values)[0]
    nnz_code = np.repeat(codes, np.diff(X.indptr))
    for k in np.unique(codes):
        m = nnz_code == k
        n_rows = int((codes == k).sum())
        df_ = np.bincount(X.indices[m], minlength=n_features)
        X.data[m] = (np.log((n_rows + 1.0) / (df_[X.indices[m]] + 1.0)) + 1.0).astype(np.float32)
    return X


def idf_pair_features(X, q, c, chunk=1_000_000):
    """IDF-weighted overlap between the token sets of q and c (same country, same IDF table)."""
    keys = ["idf_shared", "idf_jacc", "idf_cov_q", "idf_cov_c", "idf_uq_max", "idf_uc_max"]
    out = {k: np.zeros(len(q), np.float32) for k in keys}
    for i in range(0, len(q), chunk):
        A, B = X[q[i:i + chunk]], X[c[i:i + chunk]]
        Ab, Bb = A.copy(), B.copy()
        Ab.data[:] = 1.0
        Bb.data[:] = 1.0
        shA = A.multiply(Bb).tocsr()
        shB = B.multiply(Ab).tocsr()
        sa, sb = np.asarray(A.sum(1)).ravel(), np.asarray(B.sum(1)).ravel()
        sh = np.asarray(shA.sum(1)).ravel()
        uq, uc = (A - shA).tocsr(), (B - shB).tocsr()
        uq.eliminate_zeros()
        uc.eliminate_zeros()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["idf_shared"][i:i + chunk] = sh
            out["idf_jacc"][i:i + chunk] = np.nan_to_num(sh / (sa + sb - sh))
            out["idf_cov_q"][i:i + chunk] = np.nan_to_num(sh / sa)
            out["idf_cov_c"][i:i + chunk] = np.nan_to_num(sh / sb)
        out["idf_uq_max"][i:i + chunk] = uq.max(axis=1).toarray().ravel()   # rarest word only in q
        out["idf_uc_max"][i:i + chunk] = uc.max(axis=1).toarray().ravel()   # rarest word only in c
    return out


IDF_COLS = [f"{f}_{k}" for f in ("na", "ad") for k in
            ("idf_shared", "idf_jacc", "idf_cov_q", "idf_cov_c", "idf_uq_max", "idf_uc_max")]

FULL_FEATS = CHEAP_FEATS + ["p_prune", "nm_tsort", "nm_partial", "nm_jw", "nm_len_q", "nm_len_c", "core_ratio",
                            "core_tset", "core_jw", "ad_ratio", "ad_tsort", "ad_partial", "ad_len_q", "ad_len_c"] \
    + SET_COLS + IDF_COLS + ["freq_s1_name", "freq_pool_name", "pp_rank_q", "pp_gap_q", "pp_second_q"]


def name_frequencies(rec):
    key = rec["country"] + "|" + rec["name_core"]
    is1 = rec["src"].values == 1
    return {"s1": key[is1].value_counts().to_dict(), "pool": key[~is1].value_counts().to_dict()}
