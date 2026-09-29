"""Blocking miss analysis: which true pairs does candidate generation lose, and which fix recovers them?

Runs on the cached files of a finished `block` (or `probe`) stage; nothing is re-blocked:

    python -m src.diagnose --work /tmp/er/work --data-dir <dataset> --out /kaggle/working/diagnosis.txt

Sections
  A. recall overall / by country / by source / by number of true matches of the S1
  B. what missed pairs look like compared with found pairs
  C. per country, for a sample of missed pairs: rank of the true record in several retrieval views,
     and in a 2nd hop from a sibling record of the same S1 that WAS found
  D. exact blocking keys shared by the missed pair, and the size of those key blocks
  E. examples (raw text when --data-dir is given)
"""
import argparse
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .blocking import hash_vectorize, record_keys
from .config import Config

LANDMARK = re.compile(r"\b(near|opposite|behind|beside|next|adjacent|landmark|off)\b")
DIG = re.compile(r"\b\d+\b")
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def pct(x):
    return f"{100 * x:6.2f}%"


def house(addr, pc):
    d = [t for t in DIG.findall(addr) if t != pc]
    return d[0] if d else ""


# ------------------------------------------------------------------------------------------ A
def section_a(rec, gt, found, q_eval, n_cand):
    N = len(rec)
    say("== A. Recall of candidate generation ==")
    say(f"S1 queried: {len(q_eval):,}   true pairs: {len(gt):,}   found: {int(found.sum()):,}   "
        f"recall: {found.mean():.5f}")
    ctry, src = rec["country"].values, rec["src"].values
    cross = ctry[gt["q"].values] != ctry[gt["c"].values]
    say(f"true pairs whose two records have DIFFERENT country labels: {int(cross.sum()):,} "
        f"({pct(cross.mean())}); found among them: {int(found[cross].sum()):,}")
    n_true = np.bincount(gt["q"].values, minlength=N)[q_eval]
    say(f"singleton share (S1 with no true match): {pct((n_true == 0).mean())}; "
        f"mean true matches per non-singleton S1: {n_true[n_true > 0].mean():.2f}")
    t = pd.DataFrame({"country": ctry[gt["q"].values], "src": src[gt["c"].values], "found": found})
    say("recall by S1 country: " + ", ".join(f"{k}={v:.4f}" for k, v in t.groupby("country")["found"].mean().items()))
    say("recall by target source: " + ", ".join(f"S{k}={v:.4f}" for k, v in t.groupby("src")["found"].mean().items()))
    per_q = np.bincount(gt["q"].values, minlength=N)
    bucket = pd.cut(per_q[gt["q"].values], [0, 1, 2, 3, 5, 9, 10 ** 6], labels=["1", "2", "3", "4-5", "6-9", "10+"])
    say("recall by number of true matches of the S1: " + ", ".join(
        f"{k}:{v:.4f}" for k, v in t.assign(b=bucket).groupby("b", observed=True)["found"].mean().items()))
    miss_q = np.unique(gt["q"].values[~found])
    fq = np.bincount(gt["q"].values[found], minlength=N)
    has_sib = fq[gt["q"].values[~found]] > 0
    say(f"S1 with >=1 missed match: {len(miss_q):,}; of those, ALL matches missed: "
        f"{pct((fq[miss_q] == 0).mean())}")
    say(f"missed pairs whose S1 has another true match that WAS found (2nd-hop candidates): {pct(has_sib.mean())}")
    nc = n_cand[q_eval]
    say(f"candidates per S1: mean {nc.mean():.1f}; S1 with 0 candidates: {pct((nc == 0).mean())}; "
        f"S1 with misses: mean {n_cand[miss_q].mean():.1f} candidates, {per_q[miss_q].mean():.1f} true matches, "
        f"{pct((n_cand[miss_q] == 0).mean()).strip()} have 0 candidates")
    return has_sib


# ------------------------------------------------------------------------------------------ B
def pair_table(rec, qs, cs, n_jobs):
    nn, an, pc, core = rec["name_n"].values, rec["addr_n"].values, rec["postal"].values, rec["name_core"].values
    a1, a2 = nn[qs], nn[cs]
    b1, b2 = an[qs], an[cs]
    p1, p2 = pc[qs], pc[cs]
    h1 = np.array([house(x, y) for x, y in zip(b1, p1)], dtype=object)
    h2 = np.array([house(x, y) for x, y in zip(b2, p2)], dtype=object)
    d = {
        "name token_set_ratio (mean)": cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=n_jobs).mean() / 100,
        "name ratio (mean)": cpdist(a1, a2, scorer=fuzz.ratio, workers=n_jobs).mean() / 100,
        "name token_set < 0.5": (cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=n_jobs) < 50).mean(),
        "normalized names equal": np.mean(a1 == a2),
        "name cores equal": np.mean(core[qs] == core[cs]),
        "no shared name token": np.mean([not (set(x.split()) & set(y.split())) for x, y in zip(a1, a2)]),
        "addr token_set_ratio (mean)": cpdist(b1, b2, scorer=fuzz.token_set_ratio, workers=n_jobs).mean() / 100,
        "addr token_set < 0.5": (cpdist(b1, b2, scorer=fuzz.token_set_ratio, workers=n_jobs) < 50).mean(),
        "postcode: both present & equal": np.mean((p1 != "") & (p1 == p2)),
        "postcode: both present & differ": np.mean((p1 != "") & (p2 != "") & (p1 != p2)),
        "postcode: missing on target": np.mean(p2 == ""),
        "house no.: both & equal": np.mean((h1 != "") & (h1 == h2)),
        "house no.: both & differ": np.mean((h1 != "") & (h2 != "") & (h1 != h2)),
        "house no.: missing on target": np.mean(h2 == ""),
        "landmark word in target addr": np.mean([bool(LANDMARK.search(x)) for x in b2]),
        "target addr tokens (mean)": np.mean([len(x.split()) for x in b2]),
        "target name tokens (mean)": np.mean([len(x.split()) for x in a2]),
    }
    return d


def section_b(rec, gt, found, n_jobs, rng, n=20000):
    say("\n== B. Missed vs found pairs (random samples) ==")
    mi = np.flatnonzero(~found)
    fi = np.flatnonzero(found)
    mi = rng.choice(mi, min(n, len(mi)), replace=False) if len(mi) else mi
    fi = rng.choice(fi, min(n, len(fi)), replace=False)
    q, c = gt["q"].values, gt["c"].values
    tm = pair_table(rec, q[mi], c[mi], n_jobs) if len(mi) else {}
    tf = pair_table(rec, q[fi], c[fi], n_jobs)
    say(f"{'':34s} {'MISSED':>9s} {'FOUND':>9s}")
    for k in tf:
        say(f"{k:34s} {tm.get(k, float('nan')):9.3f} {tf[k]:9.3f}")


# ------------------------------------------------------------------------------------------ C
def _idf(dfc, n, max_df):
    idf = (np.log((n + 1.0) / (dfc + 1.0)) + 1.0).astype(np.float32)
    purged = dfc > max_df
    return idf, purged


def _scale(m, idf, zero_mask=None):
    m = m.tocsr(copy=True)
    w = idf if zero_mask is None else np.where(zero_mask, 0, idf).astype(np.float32)
    m.data *= w[m.indices]
    m.eliminate_zeros()
    return m


PARTS = {  # name: (column, kind, n_features)
    "nw": ("name_n", "word", 2 ** 22), "aw": ("addr_n", "word", 2 ** 22), "pw": ("postal", "word", 2 ** 20),
    "nc": ("name_n", "char", 2 ** 22), "ac": ("addr_n", "char", 2 ** 22),
}
VIEWS = {  # name: ([(part, weight)], purge)
    "tok (as run)": ([("nw", 1.0), ("aw", 0.6), ("pw", 1.2)], True),
    "tok, no purge": ([("nw", 1.0), ("aw", 0.6), ("pw", 1.2)], False),
    "addr+postcode words": ([("aw", 1.0), ("pw", 1.2)], True),
    "chr name (as run)": ([("nc", 1.0)], True),
    "chr name+addr": ([("nc", 1.0), ("ac", 0.7)], True),
}
BUCKETS = [(0, 20), (20, 40), (40, 100), (100, 500), (500, 10 ** 12)]


def section_c(rec, gt, found, has_sib, cfg, rng, n_rank, batch=16):
    say("\n== C. Rank of the missed true record in each retrieval view (sample per country) ==")
    say("rank = number of pool records scoring higher than the true record (0 = it would be first).")
    say("hop2 = query with a sibling record of the same S1 that WAS found (2nd-hop expansion).")
    q, c = gt["q"].values, gt["c"].values
    ctry, src = rec["country"].values, rec["src"].values
    fq = pd.Series(c[found], index=q[found])
    fq = fq[~fq.index.duplicated()]              # one found sibling per S1
    results = []
    for country in pd.unique(ctry[q[~found]]):
        t0 = time.time()
        sel = np.flatnonzero((~found) & (ctry[q] == country) & (ctry[c] == country))
        if not len(sel):
            continue
        sel = rng.choice(sel, min(n_rank, len(sel)), replace=False)
        pool = np.flatnonzero((src != 1) & (ctry == country))
        ppos = pd.Series(np.arange(len(pool)), index=pool)
        qq, tt = q[sel], c[sel]
        sib = fq.reindex(qq).values                       # NaN when no found sibling
        tpos = ppos.reindex(tt).values.astype(np.int64)
        has = ~np.isnan(sib)
        spos = np.full(len(sel), -1, np.int64)
        spos[has] = ppos.reindex(sib[has].astype(np.int64)).values
        n = len(pool)
        MT, sq, sqp, Qp, Qn, Sp, Sn = {}, {}, {}, {}, {}, {}, {}
        for pname, (col, kind, nf) in PARTS.items():
            M = hash_vectorize(rec[col].values[pool].tolist(), kind, nf, cfg.n_jobs, cfg.chr_ngram)
            Qraw = hash_vectorize(rec[col].values[qq].tolist(), kind, nf, 1, cfg.chr_ngram)
            dfc = np.bincount(M.indices, minlength=nf).astype(np.float64)
            idf, purged = _idf(dfc, n, cfg.tok_max_df if kind == "word" else cfg.chr_max_df)
            M = _scale(M, idf)
            M2 = M.multiply(M).tocsr()
            sq[pname] = np.asarray(M2.sum(1)).ravel()
            sqp[pname] = sq[pname] - M2 @ purged.astype(np.float32)
            Sn[pname] = M[np.maximum(spos, 0)]                          # sibling rows, no purge
            Sp[pname] = _scale(Sn[pname], np.ones_like(idf), purged)    # sibling rows, purged
            Qn[pname] = _scale(Qraw, idf)
            Qp[pname] = _scale(Qraw, idf, purged)
            MT[pname] = M.T.tocsr()
            del M, M2
        say(f"\n[{country}] pool {n:,}; sampled missed pairs {len(sel):,} "
            f"({has.sum():,} with a found sibling); views built in {time.time() - t0:.0f}s")
        ranks = {}

        def rank_rows(Qd, target, exclude, weights, purge):
            out = np.full(len(target), -1, np.int64)
            zero = np.zeros(len(target), bool)
            inv_r = 1.0 / np.sqrt(np.maximum(sum(w * w * (sqp if purge else sq)[p] for p, w in weights), 1e-12))
            for a in range(0, len(target), batch):
                b = min(a + batch, len(target))
                R = None
                for p, w in weights:
                    part = (Qd[p][0] if purge else Qd[p][1])[a:b] @ MT[p]
                    R = part * (w * w) if R is None else R + part * (w * w)
                R = R.tocsr()
                for i in range(b - a):
                    lo, hi = R.indptr[i], R.indptr[i + 1]
                    cols, vals = R.indices[lo:hi], R.data[lo:hi] * inv_r[R.indices[lo:hi]]
                    hit = cols == target[a + i]
                    st = vals[hit][0] if hit.any() else 0.0
                    zero[a + i] = st <= 0
                    better = vals > st * (1 + 1e-6) + 1e-9
                    if exclude is not None:
                        better &= cols != exclude[a + i]
                    out[a + i] = int(better.sum())
            return out, zero

        Qd = {p: (Qp[p], Qn[p]) for p in PARTS}
        Sd = {p: (Sp[p], Sn[p]) for p in PARTS}
        for vname, (weights, purge) in VIEWS.items():
            ranks[vname] = rank_rows(Qd, tpos, None, weights, purge)
        for vname in ("tok (as run)", "chr name+addr"):
            r, z = rank_rows({p: (Sd[p][0][has], Sd[p][1][has]) for p in PARTS}, tpos[has], spos[has],
                             VIEWS[vname][0], VIEWS[vname][1])
            rr, zz = np.full(len(sel), -1, np.int64), np.ones(len(sel), bool)
            rr[has], zz[has] = r, z
            ranks[f"hop2 {vname}"] = (rr, zz)
        say(f"{'view':30s} {'zero sim':>9s} " + " ".join(f"{f'<{hi}' if lo == 0 else f'{lo}-{hi - 1}' if hi < 10 ** 12 else f'>={lo}':>8s}" for lo, hi in BUCKETS))
        for vname, (r, z) in ranks.items():
            m = r >= 0 if vname.startswith("hop2") else np.ones(len(r), bool)
            m = m & (has if vname.startswith("hop2") else True)
            if not m.any():
                continue
            rz, zz = r[m], z[m]
            cells = [np.mean((~zz) & (rz >= lo) & (rz < hi)) for lo, hi in BUCKETS]
            say(f"{vname:30s} {pct(zz.mean()):>9s} " + " ".join(f"{pct(x):>8s}" for x in cells))
        rec_any = np.zeros(len(sel), bool)
        for vname, k in (("tok (as run)", 40), ("addr+postcode words", 20), ("chr name+addr", 20)):
            r, z = ranks[vname]
            rec_any |= (~z) & (r < k)
        say(f"recoverable by tok<40 | addr<20 | chr name+addr<20: {pct(rec_any.mean())}")
        r, z = ranks["hop2 tok (as run)"]
        rec_any2 = rec_any | ((r >= 0) & (~z) & (r < 10))
        say(f"... plus hop2 tok<10 from a found sibling:          {pct(rec_any2.mean())}")
        results.append((country, sel, ranks, spos, pool))
        del MT, Qp, Qn, Sp, Sn
    return results


# ------------------------------------------------------------------------------------------ D
def section_d(rec, gt, results, chunk=500_000):
    say("\n== D. Exact blocking keys shared by missed pairs (same samples) ==")
    names = {1: "name core", 2: "name+postcode", 4: "postcode+name prefix", 8: "postcode+house",
             16: "house+street+name prefix", 32: "house+street", 64: "house+rare addr words"}
    q, c = gt["q"].values, gt["c"].values
    ctry, src = rec["country"].values, rec["src"].values
    key_info = {}
    for country, sel, _, _, pool in results:
        t0 = time.time()
        pool_df = rec.iloc[pool]
        addr_freq = Counter(t for s in pool_df["addr_n"].values for t in set(s.split()))
        kq = record_keys(rec.iloc[q[sel]].reset_index(drop=True), addr_freq)
        kt = record_keys(rec.iloc[c[sel]].reset_index(drop=True), addr_freq)
        shared = kq.merge(kt, on=["pos", "key", "bit"])
        interest = set(shared["key"].tolist())
        counts = Counter()
        for a in range(0, len(pool_df), chunk):
            kp = record_keys(pool_df.iloc[a:a + chunk].reset_index(drop=True), addr_freq)
            kp = kp[kp["key"].isin(interest)]
            counts.update(kp["key"].value_counts().to_dict())
        shared["block"] = shared["key"].map(counts).fillna(0).astype(int)
        by_pos = shared.groupby("pos").agg(bits=("bit", "sum"), min_block=("block", "min"))
        n = len(sel)
        mb = by_pos["min_block"].reindex(range(n)).values
        share = ~np.isnan(mb)
        say(f"[{country}] {n:,} missed pairs; sharing >=1 exact key with the S1: {pct(share.mean())} "
            f"({time.time() - t0:.0f}s)")
        for lo, hi, lab in ((0, 11, "block <=10 (always kept)"), (11, 51, "block 11-50 (subject to union cap)"),
                            (51, 201, "block 51-200 (DROPPED: > key_max_block)"),
                            (201, 1001, "block 201-1000 (dropped)"), (1001, 10 ** 12, "block >1000 (dropped)")):
            say(f"    smallest shared block {lab:40s} {pct(np.mean(share & (mb >= lo) & (mb < hi)))}")
        fam = Counter(b for bits in by_pos["bits"] for b in names if bits & b)
        say("    shared key families: " + ", ".join(f"{names[b]}={pct(v / n).strip()}" for b, v in fam.most_common()))
        key_info[country] = (by_pos, n)
    return key_info


# ------------------------------------------------------------------------------------------ E
def load_raw(data_dir, ids):
    raw = {}
    if not data_dir:
        return raw
    for k in (1, 2, 3):
        path = os.path.join(data_dir, "train", f"train_source{k}.tsv")
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            f.readline()
            for line in f:
                i = line.split("\t", 1)[0]
                if i in ids:
                    p = line.rstrip("\r\n").split("\t")
                    raw[i] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")
    return raw


def section_e(rec, gt, results, key_info, data_dir, n_ex):
    say("\n== E. Examples of missed pairs ==")
    q, c = gt["q"].values, gt["c"].values
    ids = rec["id"].values
    names = {1: "N", 2: "NP", 4: "PP", 8: "PH", 16: "HS", 32: "HA", 64: "HR"}
    want = set()
    picks = []
    for country, sel, ranks, spos, pool in results:
        for j in range(min(n_ex, len(sel))):
            s = pool[spos[j]] if spos[j] >= 0 else None
            picks.append((country, j, q[sel[j]], c[sel[j]], s, ranks))
            want.update([ids[q[sel[j]]], ids[c[sel[j]]]] + ([ids[s]] if s is not None else []))
    raw = load_raw(data_dir, want)

    def txt(i):
        if ids[i] in raw:
            nm, ad = raw[ids[i]]
            return f"{nm} | {ad}"
        return f"{rec['name_n'].values[i]} | {rec['addr_n'].values[i]}"

    for country, j, qi, ci, si, ranks in picks:
        by_pos, _ = key_info.get(country, (pd.DataFrame(columns=["bits", "min_block"]), 0))
        kb = by_pos.loc[j] if j in by_pos.index else None
        keys = "none" if kb is None else "+".join(v for b, v in names.items() if int(kb["bits"]) & b) + f" (block {int(kb['min_block'])})"
        rk = " ".join(f"{v.split(' ')[0] if not v.startswith('hop2') else 'hop2' + v[4:8]}={('0sim' if ranks[v][1][j] else ranks[v][0][j]) if ranks[v][0][j] >= 0 else '-'}"
                      for v in ("tok (as run)", "addr+postcode words", "chr name+addr", "hop2 tok (as run)"))
        say(f"[{country}] S1     : {txt(qi)}")
        say(f"        missed : {txt(ci)}   [S{rec['src'].values[ci]}]")
        if si is not None:
            say(f"        found  : {txt(si)}")
        say(f"        ranks  : {rk} | shared keys: {keys}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--cand", default=None, help="default: cand_train.parquet (else cand_probe.parquet)")
    ap.add_argument("--qeval", default=None)
    ap.add_argument("--data-dir", default=None, help="dataset folder, for raw-text examples")
    ap.add_argument("--out", default="diagnosis.txt")
    ap.add_argument("--n-rank", type=int, default=300, help="missed pairs per country for section C/D")
    ap.add_argument("--n-examples", type=int, default=15)
    ap.add_argument("--no-keys", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = Config.from_overrides(a.set)
    t0 = time.time()
    w = a.work
    cand_path = a.cand or (os.path.join(w, "cand_train.parquet") if os.path.exists(os.path.join(w, "cand_train.parquet"))
                           else os.path.join(w, "cand_probe.parquet"))
    qeval_path = a.qeval or (os.path.join(w, "qeval_train.npy") if "cand_train" in cand_path
                             else os.path.join(w, "qeval_probe.npy"))
    say(f"diagnosing {cand_path} (queries: {qeval_path}); config max_df tok={cfg.tok_max_df} chr={cfg.chr_max_df}")
    rec = pd.read_parquet(os.path.join(w, "rec_train.parquet"))
    for col in ("id", "country", "name_n", "addr_n", "name_core", "postal"):
        rec[col] = rec[col].astype(object)
    N = len(rec)
    q_eval = np.load(qeval_path)
    in_eval = np.zeros(N, bool)
    in_eval[q_eval] = True
    gt = pd.read_parquet(os.path.join(w, "gt_train.parquet"))
    gt = gt[in_eval[gt["q"].values]].reset_index(drop=True)
    cand = pd.read_parquet(cand_path, columns=["q", "c"])
    ck = np.unique(cand["q"].values.astype(np.int64) * N + cand["c"].values.astype(np.int64))
    n_cand = np.bincount(cand["q"].values, minlength=N)
    del cand
    found = np.isin(gt["q"].values.astype(np.int64) * N + gt["c"].values.astype(np.int64), ck)
    del ck
    rng = np.random.default_rng(0)
    has_sib = section_a(rec, gt, found, q_eval, n_cand)
    section_b(rec, gt, found, cfg.n_jobs, rng)
    results = section_c(rec, gt, found, has_sib, cfg, rng, a.n_rank)
    key_info = {} if a.no_keys else section_d(rec, gt, results)
    section_e(rec, gt, results, key_info, a.data_dir, a.n_examples)
    say(f"\ndone in {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
