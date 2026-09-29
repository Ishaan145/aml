"""End-to-end pipeline: data -> normalize -> blocking -> pruning -> matching -> decision -> outputs.

Run from code/business_entity_resolution/:
    python -m src.pipeline --data-dir /path/to/student_resource/dataset --work work --out ../../output

Stages (default: all): prep, block, prune, match. Each stage caches its result in --work, so a
later stage can be re-run alone, e.g. `--stages match --set decoder=threshold`.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .blocking import generate_candidates
from .config import Config
from .decode import apply_decision, choose_decision, fit_calibrator, macro_f
from .features import CHEAP_FEATS, FULL_FEATS, add_context, block_context, cheap_features, full_features, name_frequencies
from .io_utils import load_split, log, read_ground_truth, write_id_lists
from .models import crossfit, importance
from .normalize import add_name_core, apply_mapping, mine_equivalences, normalize_frame

STR_COLS = ["id", "country", "name_n", "addr_n", "name_core", "postal"]


def _p(a, name):
    return os.path.join(a.work, name)


def load_rec(a, split):
    rec = pd.read_parquet(_p(a, f"rec_{split}.parquet"))
    for c in STR_COLS:
        rec[c] = rec[c].astype(object)
    rec["gidx"] = np.arange(len(rec), dtype=np.int32)
    return rec


# ----------------------------------------------------------------------------------------- prep
def stage_prep(a, cfg):
    """One split at a time (the real data is ~12M records per split) to bound peak memory."""
    t0 = time.time()
    os.makedirs(a.work, exist_ok=True)
    tr = load_split(a.data_dir, "train")
    log(f"normalizing train ({len(tr):,} records)")
    normalize_frame(tr, cfg.n_jobs)
    tr.drop(columns=["name", "addr"], inplace=True)
    s1_gt, gt = read_ground_truth(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"))
    pos = pd.Series(np.arange(len(tr)), index=tr["id"].values)
    pos = pos[~pos.index.duplicated()]
    qi, mi = pos.reindex(gt["s1_id"]).values, pos.reindex(gt["m_id"]).values
    ok = ~(np.isnan(qi) | np.isnan(mi))
    if (~ok).sum():
        log(f"  WARNING: {(~ok).sum():,} GT pairs reference ids missing from train files (ignored)")
    qi, mi = qi[ok].astype(np.int64), mi[ok].astype(np.int64)
    del gt, pos

    # learn token equivalences from matched pairs (names and addresses)
    rng = np.random.default_rng(cfg.seed)
    take = rng.choice(len(qi), min(len(qi), 1_500_000), replace=False)
    nn, an = tr["name_n"].values, tr["addr_n"].values
    pairs = [(nn[qi[i]], nn[mi[i]]) for i in take] + [(an[qi[i]], an[mi[i]]) for i in take]
    mapping, kept = mine_equivalences(pairs, cfg.mine_min_count, cfg.mine_max_group)
    del pairs
    log(f"  mined {len(mapping)} token equivalences; top: {kept[:15]}")

    # facts that decide pipeline branches (see eda_check.py for the full report)
    owners = pd.Series(mi).value_counts()
    ctry = tr["country"].values
    meta = {"n_gt_pairs": int(len(qi)), "s2s3_in_multiple_s1": int((owners > 1).sum()),
            "cross_country_pairs": int((ctry[qi] != ctry[mi]).sum()), "n_mined_equivalences": len(mapping)}
    meta["allow_exclusive"] = meta["s2s3_in_multiple_s1"] == 0
    log(f"  facts: {meta}")
    if meta["cross_country_pairs"] and cfg.partition_by_country:
        log("  WARNING: some true pairs cross country labels; consider --set partition_by_country=false")
    pd.DataFrame({"q": qi.astype(np.int32), "c": mi.astype(np.int32)}).to_parquet(_p(a, "gt_train.parquet"))
    json.dump({"meta": meta, "mapping": mapping}, open(_p(a, "prep_meta.json"), "w"), indent=1)

    cols = ["id", "src", "country", "name_n", "addr_n", "name_core", "postal"]
    apply_mapping(tr, mapping, cfg.n_jobs)
    add_name_core(tr, cfg.generic_top_n, cfg.generic_df_frac)
    tr[cols].to_parquet(_p(a, "rec_train.parquet"))
    del tr
    te = load_split(a.data_dir, "test")
    log(f"normalizing test ({len(te):,} records)")
    normalize_frame(te, cfg.n_jobs)
    te.drop(columns=["name", "addr"], inplace=True)
    apply_mapping(te, mapping, cfg.n_jobs)
    add_name_core(te, cfg.generic_top_n, cfg.generic_df_frac)
    te[cols].to_parquet(_p(a, "rec_test.parquet"))
    log(f"prep done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------------------- block
def _gt_keys(a, n):
    gt = pd.read_parquet(_p(a, "gt_train.parquet"))
    return gt, gt["q"].values.astype(np.int64) * n + gt["c"].values.astype(np.int64)


def stage_block(a, cfg):
    for split in ("train", "test"):
        t0 = time.time()
        rec = load_rec(a, split)
        log(f"blocking {split}")
        cand = block_context(generate_candidates(rec, cfg))       # every S1 queried: true competition
        is_s1 = rec["src"].values == 1
        if split == "train" and a.train_frac < 1.0:               # subsample AFTER competition features
            keep = np.random.default_rng(cfg.seed).random(len(rec)) < a.train_frac
            is_s1 &= keep
            cand = cand[keep[cand["q"].values]].reset_index(drop=True)
        q_eval = rec["gidx"].values[is_s1]
        if split == "train":
            gt, keys = _gt_keys(a, len(rec))
            ck = cand["q"].values.astype(np.int64) * len(rec) + cand["c"].values.astype(np.int64)
            cand["label"] = np.isin(ck, keys).astype(np.int8)
            n_gt = int(np.isin(gt["q"].values, q_eval).sum())
            log(f"  blocking recall (pair completeness) = {cand['label'].sum() / max(n_gt, 1):.5f} "
                f"({int(cand['label'].sum()):,}/{n_gt:,}); candidates/S1 = {len(cand) / max(len(q_eval), 1):.2f}")
            # sample of true pairs that blocking missed, for error analysis (work/missed_pairs.tsv)
            hit = np.isin(keys, ck)
            miss = gt[~hit & np.isin(gt["q"].values, q_eval)]
            miss = miss.sample(min(300, len(miss)), random_state=0)
            pd.DataFrame({"s1_id": rec["id"].values[miss["q"].values], "s1_name": rec["name_n"].values[miss["q"].values],
                          "s1_addr": rec["addr_n"].values[miss["q"].values],
                          "m_id": rec["id"].values[miss["c"].values], "m_name": rec["name_n"].values[miss["c"].values],
                          "m_addr": rec["addr_n"].values[miss["c"].values]}).to_csv(
                _p(a, "missed_pairs.tsv"), sep="\t", index=False)
        np.save(_p(a, f"qeval_{split}.npy"), q_eval)
        cand.to_parquet(_p(a, f"cand_{split}.parquet"))
        log(f"block {split} done in {time.time() - t0:.0f}s")


def stage_probe(a, cfg):
    """Fast blocking diagnostic: query a sample of train S1s against the FULL pool, report recall
    per view, candidates/S1 and time. Use it to tune blocking knobs on the real data in minutes."""
    t0 = time.time()
    rec = load_rec(a, "train")
    mask = np.random.default_rng(cfg.seed + 1).random(len(rec)) < a.probe_frac
    cand = generate_candidates(rec, cfg, mask)
    q_eval = rec["gidx"].values[(rec["src"].values == 1) & mask]
    gt, keys = _gt_keys(a, len(rec))
    gt = gt[np.isin(gt["q"].values, q_eval)]
    ck = cand["q"].values.astype(np.int64) * len(rec) + cand["c"].values.astype(np.int64)
    lab = np.isin(ck, keys)
    n_gt = max(len(gt), 1)
    rep = {"probe_s1": int(len(q_eval)), "gt_pairs": int(len(gt)), "recall_union": round(lab.sum() / n_gt, 5),
           "recall_tok": round((lab & (cand["score_tok"].values > 0)).sum() / n_gt, 5),
           "recall_chr": round((lab & (cand["score_chr"].values > 0)).sum() / n_gt, 5),
           "recall_key": round((lab & (cand["key_bits"].values > 0)).sum() / n_gt, 5),
           "only_key": int((lab & (cand["n_views"].values == 1) & (cand["key_bits"].values > 0)).sum()),
           "only_tok": int((lab & (cand["n_views"].values == 1) & (cand["score_tok"].values > 0)).sum()),
           "only_chr": int((lab & (cand["n_views"].values == 1) & (cand["score_chr"].values > 0)).sum()),
           "recall_rev": round((lab & (cand["score_rev"].values > 0)).sum() / n_gt, 5),
           "only_rev": int((lab & (cand["n_views"].values == 1) & (cand["score_rev"].values > 0)).sum()),
           "cands_per_s1": round(len(cand) / max(len(q_eval), 1), 2), "seconds": round(time.time() - t0, 1)}
    log(f"PROBE: {json.dumps(rep)}")
    log(f"  full-train blocking time ~ {rep['seconds'] / max(a.probe_frac, 1e-9):.0f}s (rough linear extrapolation)")


# ---------------------------------------------------------------------------------------- prune
def _write_candidates(a, rec, df, name, col):
    s1_ids = list(dict.fromkeys(rec["id"].values[rec["src"].values == 1]))
    write_id_lists(os.path.join(a.out, name), s1_ids, rec["id"].values[df["q"].values],
                   rec["id"].values[df["c"].values], col)


def _reduction_stats(rec, df, q_eval):
    """Reduction ratio vs. comparing every S1 with every same-country S2/S3 record."""
    src, ctry = rec["src"].values, rec["country"].values
    pool = pd.Series(ctry[src != 1]).value_counts()
    full = float(pd.Series(ctry[q_eval]).map(pool).fillna(0).sum())
    return {"pairs": int(len(df)), "per_s1_mean": round(len(df) / max(len(q_eval), 1), 3),
            "per_s1_p95": float(np.percentile(np.bincount(df["q"].values, minlength=len(rec))[q_eval], 95)) if len(q_eval) else 0.0,
            "s1_with_zero": int((np.bincount(df["q"].values, minlength=len(rec))[q_eval] == 0).sum()),
            "reduction_ratio": round(1 - len(df) / full, 8) if full else None}


def stage_prune(a, cfg):
    """Train split first (fit pruner, choose cut-off), then test; frees memory in between."""
    t0 = time.time()
    rtr = load_rec(a, "train")
    tr = pd.read_parquet(_p(a, "cand_train.parquet"))
    log(f"cheap features (train, {len(tr):,} pairs)")
    tr = cheap_features(tr, rtr, cfg.n_jobs)
    X, y = tr[CHEAP_FEATS].values.astype(np.float32), tr["label"].values
    oof, rounds, model = crossfit(X, y, tr["q"].values, cfg, name="pruner")
    del X
    tr["p_prune"] = oof
    tau = float(np.quantile(oof[y == 1], cfg.prune_recall_loss)) if y.sum() else 0.0
    tau_train = float(np.quantile(oof[y == 1], cfg.prune_recall_loss / 10)) if y.sum() else 0.0
    log(f"  pruning threshold tau={tau:.5f} (keeps {1 - cfg.prune_recall_loss:.1%} of blocked true pairs), "
        f"cap {cfg.prune_max}/S1; matcher trains on the looser tau={tau_train:.5f} to see more hard negatives")

    def prune(df, t):
        # within-S1 context of the pruner score over the whole blocked set of that S1
        df = add_context(df, "p_prune", "pp", reverse=False)
        df = df[df["p_prune"] >= t].sort_values(["q", "p_prune"], ascending=[True, False])
        df = df[df.groupby("q").cumcount() < cfg.prune_max].reset_index(drop=True)
        df["in_cand"] = (df["p_prune"].values >= tau).astype(np.int8)
        return df

    tr = prune(tr, min(tau, tau_train))
    qe_tr = np.load(_p(a, "qeval_train.npy"))
    gt = pd.read_parquet(_p(a, "gt_train.parquet"))
    n_gt = int(np.isin(gt["q"].values, qe_tr).sum())
    strict = tr[tr["in_cand"] == 1]
    stats = {"train_pair_completeness_after_prune": round(strict["label"].sum() / max(n_gt, 1), 6),
             "train": _reduction_stats(rtr, strict, qe_tr), "train_rows_for_matcher": int(len(tr)),
             "prune_tau": tau, "pruner_rounds": rounds, "pruner_importance": importance(model, CHEAP_FEATS, 12)}
    tr.to_parquet(_p(a, "pr_train.parquet"))
    del tr, strict, rtr, oof, y

    rte = load_rec(a, "test")
    te = pd.read_parquet(_p(a, "cand_test.parquet"))
    log(f"cheap features (test, {len(te):,} pairs)")
    te = cheap_features(te, rte, cfg.n_jobs)
    te["p_prune"] = model.predict(te[CHEAP_FEATS].values.astype(np.float32)).astype(np.float32)
    te = prune(te, tau)
    stats["test"] = _reduction_stats(rte, te, np.load(_p(a, "qeval_test.npy")))
    log(f"  after pruning: {json.dumps(stats)}")
    json.dump(stats, open(_p(a, "prune_report.json"), "w"), indent=1)
    te.to_parquet(_p(a, "pr_test.parquet"))
    _write_candidates(a, rte, te, "candidate_pairs.tsv", "candidate_entity_ids")
    log(f"prune done in {time.time() - t0:.0f}s; wrote {os.path.join(a.out, 'candidate_pairs.tsv')}")


# ---------------------------------------------------------------------------------------- match
CE_FEATS = ["ce", "ce_rank_q", "ce_gap_q", "ce_second_q", "ce_rank_c", "ce_gap_c", "ce_mutual"]


def add_ce(df, a, split):
    """Merge optional cross-encoder scores (work/ce_<split>.parquet from src.ce_model score) and their
    listwise context: rank/gap among the S1's candidates and among the S1s competing for the record."""
    path = _p(a, f"ce_{split}.parquet")
    if not os.path.exists(path):
        return df, False
    ce = pd.read_parquet(path, columns=["q", "c", "ce"])
    df = df.merge(ce, on=["q", "c"], how="left")
    miss = float(df["ce"].isna().mean())
    log(f"  cross-encoder scores merged for {split}: {1 - miss:.2%} of pairs covered")
    if miss > 0.01:
        raise RuntimeError(f"ce_{split}.parquet covers only {1 - miss:.2%} of pairs: wrong work folder?")
    df["ce"] = df["ce"].fillna(df["ce"].min())
    return add_context(df, "ce", "ce"), True


def stage_match(a, cfg):
    t0 = time.time()
    rtr = load_rec(a, "train")
    tr = pd.read_parquet(_p(a, "pr_train.parquet"))
    log(f"full features (train, {len(tr):,} pairs)")
    tr = full_features(tr, rtr, name_frequencies(rtr), cfg.n_jobs)
    tr, use_ce = add_ce(tr, a, "train")
    feats = FULL_FEATS + (CE_FEATS if use_ce else [])
    if use_ce and not os.path.exists(_p(a, "ce_test.parquet")):
        raise RuntimeError("ce_train.parquet found but ce_test.parquet missing")
    X, y = tr[feats].values.astype(np.float32), tr["label"].values
    oof, rounds, model = crossfit(X, y, tr["q"].values, cfg, name="matcher")
    del X
    strict = tr["in_cand"].values == 1
    cal = fit_calibrator(oof[strict], y[strict])
    tr["p"] = cal.predict(oof).astype(np.float32)
    tr = tr[tr["in_cand"] == 1].reset_index(drop=True)   # evaluate exactly what test inference sees

    meta = json.load(open(_p(a, "prep_meta.json")))["meta"]
    q_eval = np.load(_p(a, "qeval_train.npy"))
    gt = pd.read_parquet(_p(a, "gt_train.parquet"))
    n_true = np.bincount(gt["q"].values, minlength=len(rtr))[q_eval]
    choice, table = choose_decision(tr, "p", q_eval, n_true, cfg, meta["allow_exclusive"])
    log("  decision table (validation = out-of-fold train):\n" + table.to_string(index=False))
    sel_tr = apply_decision(tr, "p", choice, cfg.beta)
    f, per = macro_f(q_eval, n_true, tr["q"].values[sel_tr], tr["label"].values[sel_tr].astype(bool), cfg.beta)
    ctry = rtr["country"].values[q_eval]
    sing = n_true == 0
    report = {"validation_macro_f05": round(f, 6),
              "by_country": {c: round(float(per[ctry == c].mean()), 6) for c in pd.unique(ctry)},
              "singleton_share": round(float(sing.mean()), 4),
              "singleton_accuracy": round(float(per[sing].mean()), 6) if sing.any() else None,
              "non_singleton_f05": round(float(per[~sing].mean()), 6) if (~sing).any() else None,
              "choice": {k: (v.item() if hasattr(v, "item") else v) for k, v in choice.items()},
              "decision_table": table.to_dict("records"),
              "matcher_rounds": rounds, "matcher_importance": importance(model, feats, 25),
              "cross_encoder": use_ce,
              "note": "decision parameters are tuned on the same out-of-fold predictions; expect a small optimism"}
    log(f"  VALIDATION: {json.dumps(report, default=str)}")
    sfx = "_ce" if use_ce else ""                          # never overwrite the baseline run's files
    tr[["q", "c", "label", "p"]].assign(selected=sel_tr).to_parquet(_p(a, f"oof_train{sfx}.parquet"))
    model.save_model(_p(a, f"matcher{sfx}.txt"))
    del tr, rtr, oof, y

    rte = load_rec(a, "test")
    te = pd.read_parquet(_p(a, "pr_test.parquet"))
    log(f"full features (test, {len(te):,} pairs)")
    te = full_features(te, rte, name_frequencies(rte), cfg.n_jobs)
    if use_ce:
        te, _ = add_ce(te, a, "test")
    te["p"] = cal.predict(model.predict(te[feats].values.astype(np.float32))).astype(np.float32)
    sel_te = apply_decision(te, "p", choice, cfg.beta)
    out = te[sel_te]
    write_id_lists(os.path.join(a.out, "matching_results.tsv"),
                   list(dict.fromkeys(rte["id"].values[rte["src"].values == 1])),
                   rte["id"].values[out["q"].values], rte["id"].values[out["c"].values], "matched_entity_ids")
    report["test_predicted_pairs_by_country"] = pd.Series(rte["country"].values[out["q"].values]).value_counts().to_dict()
    report["test_s1_with_matches"] = int(out["q"].nunique())
    json.dump(report, open(_p(a, f"match_report{sfx}.json"), "w"), indent=1, default=str)
    log(f"match done in {time.time() - t0:.0f}s; wrote {os.path.join(a.out, 'matching_results.tsv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--work", default="work", help="cache folder for intermediate files")
    ap.add_argument("--out", default="../../output", help="where the two submission TSVs go")
    ap.add_argument("--stages", default="prep,block,prune,match")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="share of train S1 entities used as queries (e.g. 0.2 for a fast dev loop)")
    ap.add_argument("--probe-frac", type=float, default=0.05, help="share of train S1s used by the probe stage")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set n_jobs=8 tok_topk=30")
    a = ap.parse_args()
    cfg = Config.from_overrides(a.set)
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    log(f"config: {cfg}")
    stages = {"prep": stage_prep, "probe": stage_probe, "block": stage_block, "prune": stage_prune,
              "match": stage_match}
    for s in a.stages.split(","):
        stages[s.strip()](a, cfg)


if __name__ == "__main__":
    main()
