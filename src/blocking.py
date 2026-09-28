"""Candidate generation (blocking), run per country partition.

Three complementary views, each a sparse top-k retrieval S1 -> (S2 u S3):
  tok : word tokens of name + address + postcode, IDF-weighted (Sparkly-style sparse retrieval),
        with block purging (tokens in > tok_max_df pool records are dropped)
  chr : char n-grams of the name core (typo-robust)
  rev : reverse retrieval: each S2/S3 record -> its top-k S1s (S1 is deduplicated, so this is the
        high-recall direction when an S1 has many matches)
  key : exact standard-blocking keys (sorted name core, name+postcode, postcode+house number, ...)
Union -> reciprocal rank fusion -> keep top `union_cap` per S1 (+ forced specific-key / reverse hits).
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .io_utils import log

_DIGIT_TOKEN = __import__("re").compile(r"\b\d+\b")


def _split(s):
    return s.split()


def _hash_chunk(texts, kind, n_features, ngram):
    if kind == "word":
        hv = HashingVectorizer(analyzer=_split, n_features=n_features, alternate_sign=False,
                               norm=None, binary=True, dtype=np.float32)
    else:
        hv = HashingVectorizer(analyzer="char_wb", ngram_range=(ngram, ngram), n_features=n_features,
                               alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    return hv.transform(texts)


def hash_vectorize(texts, kind, n_features, n_jobs, ngram=4, chunk=250_000):
    if len(texts) <= chunk or n_jobs <= 1:
        return _hash_chunk(texts, kind, n_features, ngram).tocsr()
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_hash_chunk)(texts[i:i + chunk], kind, n_features, ngram) for i in range(0, len(texts), chunk))
    return sp.vstack(parts).tocsr()


def idf_weights(pool_mat, max_df):
    n = pool_mat.shape[0]
    df = np.bincount(pool_mat.indices, minlength=pool_mat.shape[1]).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    idf[df > max_df] = 0.0          # block purging: too common to discriminate
    return idf.astype(np.float32)


def _scale_cols(m, w):
    m = m.tocsr(copy=True)
    m.data *= w[m.indices]
    m.eliminate_zeros()
    return m


def choose_max_df(q_mats, dfs, cap, budget, sample=20_000, seed=0):
    """Cost-budgeted block purging. Retrieval cost per query = sum of posting-list lengths of its
    tokens. Pick the largest document-frequency cut-off (<= cap) whose estimated mean cost per query
    stays within `budget`, so blocking time is bounded whatever the data looks like."""
    n = q_mats[0].shape[0]
    rows = np.random.default_rng(seed).choice(n, min(n, sample), replace=False) if n > sample else np.arange(n)
    qs = [(q[rows] > 0).astype(np.float32) for q in q_mats]
    m = float(cap)
    while True:
        cost = sum(float((q @ np.where(df <= m, df, 0).astype(np.float32)).mean()) for q, df in zip(qs, dfs))
        if cost <= budget or m <= 50:
            return int(m), cost
        m *= 0.7


def build_view(pool, qry, view, cfg):
    """Return (Q, P^T, info) for one view; rows L2-normalized, tokens purged to fit the cost budget."""
    if view == "tok":
        specs, kind = [("name_n", cfg.w_name, 2 ** 22), ("addr_n", cfg.w_addr, 2 ** 22),
                       ("postal", cfg.w_postal, 2 ** 20)], "word"
        cap, budget = cfg.tok_max_df, cfg.tok_budget
    else:
        specs, kind = [("name_n", 1.0, 2 ** 22)], "char"
        cap, budget = cfg.chr_max_df, cfg.chr_budget
    ps, qs, dfs = [], [], []
    for col, w, nf in specs:
        p = hash_vectorize(pool[col].tolist(), kind, nf, cfg.n_jobs, cfg.chr_ngram)
        ps.append(p)
        qs.append(hash_vectorize(qry[col].tolist(), kind, nf, cfg.n_jobs, cfg.chr_ngram))
        dfs.append(np.bincount(p.indices, minlength=nf).astype(np.float64))
    max_df, cost = choose_max_df(qs, dfs, cap, budget)
    n = ps[0].shape[0]
    mats_p, mats_q = [], []
    for (col, w, nf), p, q, df in zip(specs, ps, qs, dfs):
        idf = (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32) * w
        idf[df > max_df] = 0.0          # block purging: too common to discriminate / too costly
        mats_p.append(_scale_cols(p, idf))
        mats_q.append(_scale_cols(q, idf))
    P = normalize(sp.hstack(mats_p).tocsr(), norm="l2", copy=False).astype(np.float32)
    Q = normalize(sp.hstack(mats_q).tocsr(), norm="l2", copy=False).astype(np.float32)
    return Q, P.T.tocsr(), {"max_df": max_df, "est_postings_per_query": round(cost)}


def topk(Qc, PT, k, thr, n_jobs):
    """Top-k cosine per query row. Returns (row, col, score, rank) arrays."""
    if Qc.nnz == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32), e
    R = sp_matmul_topn(Qc, PT, top_n=k, threshold=thr, sort=True, n_threads=n_jobs)
    counts = np.diff(R.indptr)
    rows = np.repeat(np.arange(R.shape[0]), counts)
    ranks = np.arange(R.nnz) - np.repeat(R.indptr[:-1], counts)
    return rows, R.indices.astype(np.int64), R.data.astype(np.float32), ranks


def record_keys(df, addr_freq):
    """Exact standard-blocking keys; each key family is one bit in key_bits.
    Address keys use both the first two address words and the two rarest words among the first six
    (rarity from this country's own data): the first is robust to typos in rare words, the second
    keys 'boulevard de la X' on X rather than on 'boulevard de'."""
    core, name = df["name_core"].values, df["name_n"].values
    postal, addr = df["postal"].values, df["addr_n"].values
    out = []  # (row_pos, key_hash, bit)
    for i in range(len(df)):
        ctoks = sorted(core[i].split())
        ntoks = name[i].split()
        k = " ".join(ctoks)
        pre = ntoks[0][:3] if ntoks else ""
        pc = postal[i]
        digs = _DIGIT_TOKEN.findall(addr[i])
        house = next((d for d in digs if d != pc), "")
        words = [t for t in addr[i].split()[:6] if not t.isdigit()]
        street = words[:2]                                                    # first two words
        rare = sorted(sorted(words, key=lambda t: addr_freq.get(t, 0))[:2])   # two rarest words
        if len(k) >= 3:
            out.append((i, hash("n|" + k), 1))                                  # same name core
            if pc:
                out.append((i, hash("np|" + k + "|" + pc), 2))
        if pc:
            out.append((i, hash("pp|" + pc + "|" + pre), 4))                    # postcode + name prefix
            if house:
                out.append((i, hash("ph|" + pc + "|" + house), 8))              # same address, any name (DBA)
        if house and street:
            s2 = " ".join(street)
            out.append((i, hash("hs|" + house + "|" + s2 + "|" + pre), 16))   # house+street+name prefix
            out.append((i, hash("ha|" + house + "|" + s2), 32))                # house+street, any name (DBA)
            if rare != street:
                out.append((i, hash("hr|" + house + "|" + " ".join(rare)), 64))  # house+rarest words (any name)
    # keys are 64-bit hashes (only compared within this process): ~10x less memory than strings
    if not out:
        return pd.DataFrame({"pos": np.array([], np.int64), "key": np.array([], np.int64), "bit": np.array([], np.int16)})
    pos, key, bit = zip(*out)
    return pd.DataFrame({"pos": np.array(pos, np.int64), "key": np.array(key, np.int64), "bit": np.array(bit, np.int16)})


def key_view(pool, qry, cfg):
    from collections import Counter
    addr_freq = Counter(t for s in pool["addr_n"].values for t in set(s.split()))
    pk, qk = record_keys(pool, addr_freq), record_keys(qry, addr_freq)
    pk["block"] = pk.groupby("key")["pos"].transform("size").astype(np.int32)
    pk = pk[pk["block"] <= cfg.key_max_block]
    m = qk.merge(pk, on=["key", "bit"], suffixes=("_q", "_p"))
    if m.empty:
        return pd.DataFrame({"qpos": np.array([], np.int64), "ppos": np.array([], np.int64),
                             "key_bits": np.array([], np.int16), "key_block": np.array([], np.int32)})
    m = m.groupby(["pos_q", "pos_p"], sort=False).agg(key_bits=("bit", "sum"), key_block=("block", "min")).reset_index()
    m.columns = ["qpos", "ppos", "key_bits", "key_block"]
    return m


def _union(tok, chr_, rev, kv, cap, keep_block, rev_keep):
    df = (tok.merge(chr_, on=["qpos", "ppos"], how="outer").merge(rev, on=["qpos", "ppos"], how="outer")
          .merge(kv, on=["qpos", "ppos"], how="outer"))
    for v in ("tok", "chr", "rev"):
        df[f"score_{v}"] = df[f"score_{v}"].fillna(0).astype(np.float32)
        df[f"rank_{v}"] = df[f"rank_{v}"].fillna(99).astype(np.float32)
    df["rev_gap"] = df["rev_gap"].fillna(1.0).astype(np.float32)
    df["key_bits"] = df["key_bits"].fillna(0).astype(np.int16)
    df["key_block"] = df["key_block"].fillna(9999).astype(np.float32)
    hit_t, hit_c, hit_r, hit_k = (df["score_tok"] > 0, df["score_chr"] > 0, df["score_rev"] > 0,
                                  df["key_bits"] > 0)
    df["n_views"] = (hit_t.astype(np.int8) + hit_c.astype(np.int8) + hit_r.astype(np.int8)
                     + hit_k.astype(np.int8)).astype(np.int8)
    # key hits count as a rank that depends on block size: a key shared by 2 records is near-proof
    key_rank = np.log2(1.0 + df["key_block"].values)
    df["rrf"] = (hit_t / (61 + df["rank_tok"]) + hit_c / (61 + df["rank_chr"]) + hit_r / (61 + df["rank_rev"])
                 + hit_k / (61.0 + key_rank)).astype(np.float32)
    # always kept: very specific key hits, and S1s that are a pool record's top reverse choices
    forced = (df["key_block"].values <= keep_block) | (df["rank_rev"].values < rev_keep)
    df = df.assign(_f=forced).sort_values(["qpos", "_f", "rrf", "score_tok"], ascending=[True, False, False, False])
    keep = (df.groupby("qpos").cumcount().values < cap) | df["_f"].values
    return df[keep].drop(columns="_f")


def reverse_view(pool, s1_all, qry, cfg):
    """Reverse retrieval: every S2/S3 record queries the index of ALL S1 records of the country and
    keeps its top `rev_topk` S1s. Source 1 is deduplicated and each S2/S3 record belongs to at most
    one S1, so the true owner is almost always among the few best S1s for that record, even when the
    S1 has many matches and many look-alike pool records competing for the forward top-k."""
    t0 = time.time()
    Q, PT, info = build_view(s1_all, pool, "tok", cfg)          # index = S1, queries = pool records
    log(f"    rev view ready ({time.time() - t0:.0f}s): {info}")
    # map S1-index positions -> positions in qry (probe queries only a subset of S1)
    qpos_of = pd.Series(np.arange(len(qry)), index=qry["gidx"].values)
    s1_to_q = qpos_of.reindex(s1_all["gidx"].values).fillna(-1).values.astype(np.int64)
    parts, chunk = [], cfg.query_chunk * 2
    n_chunks, t1 = -(-len(pool) // chunk), time.time()
    for i, a in enumerate(range(0, len(pool), chunk)):
        b = min(a + chunk, len(pool))
        r, c, sc, rk = topk(Q[a:b], PT, cfg.rev_topk, cfg.min_sim, cfg.n_jobs)
        top = np.zeros(b - a, np.float32)
        first = rk == 0
        top[r[first]] = sc[first]
        qp = s1_to_q[c]
        ok = qp >= 0
        parts.append(pd.DataFrame({"qpos": qp[ok], "ppos": (r + a)[ok], "score_rev": sc[ok],
                                   "rank_rev": rk[ok].astype(np.float32), "rev_gap": (top[r] - sc)[ok]}))
        if (i + 1) % 10 == 0 or i + 1 == n_chunks:
            el = time.time() - t1
            log(f"    rev chunk {i + 1}/{n_chunks}: {el:.0f}s elapsed, ETA {el / (i + 1) * (n_chunks - i - 1):.0f}s")
    return pd.concat(parts, ignore_index=True).sort_values("qpos", kind="stable")


def block_partition(pool, qry, cfg, s1_all=None):
    """All candidates for one country. pool/qry are record frames with a 'gidx' column; s1_all is
    every S1 of the country (the reverse index must hold all S1s even when only a subset is queried).
    Indexes are built once; queries are processed in chunks, with progress + ETA logs."""
    s1_all = qry if s1_all is None else s1_all
    t0 = time.time()
    Qt, PTt, it = build_view(pool, qry, "tok", cfg)
    log(f"    tok view ready ({time.time() - t0:.0f}s): {it}")
    use_chr = cfg.chr_topk > 0
    if use_chr:
        Qc, PTc, ic = build_view(pool, qry, "chr", cfg)
        log(f"    chr view ready ({time.time() - t0:.0f}s): {ic}")
    rv = reverse_view(pool, s1_all, qry, cfg) if cfg.rev_topk > 0 else pd.DataFrame(
        {"qpos": np.array([], np.int64), "ppos": np.array([], np.int64), "score_rev": np.array([], np.float32),
         "rank_rev": np.array([], np.float32), "rev_gap": np.array([], np.float32)})
    rq = rv["qpos"].values
    kv = key_view(pool, qry, cfg).sort_values("qpos")
    kq = kv["qpos"].values
    log(f"    key view ready ({time.time() - t0:.0f}s): {len(kv):,} key pairs, {len(rv):,} reverse pairs")
    out, n_chunks, t1 = [], -(-len(qry) // cfg.query_chunk), time.time()
    empty_chr = pd.DataFrame({"qpos": np.array([], np.int64), "ppos": np.array([], np.int64),
                              "score_chr": np.array([], np.float32), "rank_chr": np.array([], np.float32)})
    for i, a in enumerate(range(0, len(qry), cfg.query_chunk)):
        b = min(a + cfg.query_chunk, len(qry))
        r, c, s, rk = topk(Qt[a:b], PTt, cfg.tok_topk, cfg.min_sim, cfg.n_jobs)
        tok = pd.DataFrame({"qpos": r + a, "ppos": c, "score_tok": s, "rank_tok": rk.astype(np.float32)})
        if use_chr:
            r, c, s, rk = topk(Qc[a:b], PTc, cfg.chr_topk, cfg.min_sim, cfg.n_jobs)
            chr_ = pd.DataFrame({"qpos": r + a, "ppos": c, "score_chr": s, "rank_chr": rk.astype(np.float32)})
        else:
            chr_ = empty_chr
        lo, hi = np.searchsorted(kq, a), np.searchsorted(kq, b)
        rlo, rhi = np.searchsorted(rq, a), np.searchsorted(rq, b)
        out.append(_union(tok, chr_, rv.iloc[rlo:rhi], kv.iloc[lo:hi], cfg.union_cap, cfg.key_keep_block,
                          cfg.rev_keep_rank))
        el = time.time() - t1
        log(f"    chunk {i + 1}/{n_chunks}: {el:.0f}s elapsed, ETA {el / (i + 1) * (n_chunks - i - 1):.0f}s")
    df = pd.concat(out, ignore_index=True)
    df["q"] = qry["gidx"].values[df["qpos"].values].astype(np.int32)
    df["c"] = pool["gidx"].values[df["ppos"].values].astype(np.int32)
    return df.drop(columns=["qpos", "ppos"]).reset_index(drop=True)


def generate_candidates(rec, cfg, query_mask=None):
    # query_mask limits which S1 records are queried (used by the fast 'probe' stage)
    """rec: all records of one split (with gidx). Returns candidate frame for S1 queries."""
    is_q = rec["src"].values == 1
    if query_mask is not None:
        is_q &= query_mask
    part_key = rec["country"].values if cfg.partition_by_country else np.full(len(rec), "ALL", dtype=object)
    out = []
    for c in pd.unique(part_key[is_q]):
        in_c = part_key == c
        qry = rec[is_q & in_c]
        s1_all = rec[(rec["src"].values == 1) & in_c]
        pool = rec[(rec["src"].values != 1) & in_c]
        if len(pool) == 0 or len(qry) == 0:
            log(f"  country={c!r}: {len(qry):,} S1, {len(pool):,} pool -> no candidates possible")
            continue
        log(f"  country={c!r}: {len(qry):,} S1 vs {len(pool):,} pool -> building indexes")
        df = block_partition(pool, qry, cfg, s1_all)
        log(f"  country={c!r}: {len(qry):,} S1 vs {len(pool):,} pool -> {len(df):,} candidates "
            f"({len(df) / len(qry):.1f}/S1)")
        out.append(df)
    cols = ["q", "c", "score_tok", "rank_tok", "score_chr", "rank_chr", "score_rev", "rank_rev", "rev_gap",
            "key_bits", "key_block", "n_views", "rrf"]
    if not out:
        return pd.DataFrame({c: np.array([], np.float32) for c in cols}).astype({"q": np.int32, "c": np.int32})
    return pd.concat(out, ignore_index=True)[cols]
