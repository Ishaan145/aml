"""Decision layer: calibration, S2/S3 exclusivity, and per-entity set selection for macro F0.5.

Two decoders are compared on out-of-fold predictions and the better one is used:
  threshold  : predict every candidate with p >= t (t searched on validation)
  expected_f : for each S1, pick the top-k (k = 0..n) that maximises the expected F0.5 under
               independent calibrated match probabilities; k = 0 ("no match") has expected
               score P(no candidate is a match). Motivated by Dembczynski et al. (NeurIPS 2011):
               the F-optimal set is generally not a fixed threshold on marginals.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(p, y, max_rows=3_000_000, seed=0):
    if len(p) > max_rows:
        idx = np.random.default_rng(seed).choice(len(p), max_rows, replace=False)
        p, y = p[idx], y[idx]
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def exclusive_mask(df, pcol):
    """Keep, for each S2/S3 record, only its best-scoring S1 (ties kept)."""
    best = df.groupby("c")[pcol].transform("max").values
    return df[pcol].values >= best


def macro_f(q_eval, n_true_eval, sel_q, sel_true, beta=0.5):
    """Exact macro F_beta over the S1 ids in q_eval (arrays of ints), singleton rule included."""
    size = int(max(q_eval.max(initial=0), sel_q.max(initial=0))) + 1
    npred = np.bincount(sel_q, minlength=size)[q_eval].astype(np.float64)
    tp = np.bincount(sel_q, weights=sel_true.astype(np.float64), minlength=size)[q_eval]
    nt = n_true_eval.astype(np.float64)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(tp > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    f = np.where(nt == 0, (npred == 0).astype(np.float64), f)
    return float(f.mean()), f


def expected_f_select(q, p, beta=0.5, batch=50_000):
    """Boolean selection mask (aligned with inputs) maximising expected F_beta per S1."""
    n_rows = len(q)
    sel = np.zeros(n_rows, dtype=bool)
    if n_rows == 0:
        return sel
    order = np.lexsort((-p, q))
    qs, ps = q[order], p[order].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    sizes = np.diff(np.r_[starts, n_rows])
    b2 = beta * beta
    for n in np.unique(sizes):
        st_all = starts[sizes == n]
        for g0 in range(0, len(st_all), batch):
            st = st_all[g0:g0 + batch]
            _select_block(st, n, ps, order, sel, b2)
    return sel


def _select_block(st, n, ps, order, sel, b2):
    P = ps[st[:, None] + np.arange(n)[None, :]]                # G x n, sorted desc
    G = P.shape[0]
    pre = [np.ones((G, 1))]
    for j in range(n):
        prev = pre[-1]
        nxt = np.zeros((G, j + 2))
        nxt[:, :-1] += prev * (1 - P[:, j:j + 1])
        nxt[:, 1:] += prev * P[:, j:j + 1]
        pre.append(nxt)
    suf = [None] * (n + 1)
    suf[n] = np.ones((G, 1))
    for j in range(n - 1, -1, -1):
        prev = suf[j + 1]
        nxt = np.zeros((G, prev.shape[1] + 1))
        nxt[:, :-1] += prev * (1 - P[:, j:j + 1])
        nxt[:, 1:] += prev * P[:, j:j + 1]
        suf[j] = nxt
    best_val = suf[0][:, 0].copy()                              # k = 0: P(no match among candidates)
    best_k = np.zeros(G, dtype=np.int64)
    for k in range(1, n + 1):
        a = np.arange(k + 1)[:, None]
        b = np.arange(n - k + 1)[None, :]
        W = (1 + b2) * a / (b2 * (a + b) + k)                    # F when a of k picked are true, b missed
        val = np.einsum("ga,ab,gb->g", pre[k], W, suf[k])
        better = val > best_val
        best_val[better] = val[better]
        best_k[better] = k
    pick = np.arange(n)[None, :] < best_k[:, None]
    sel[order[(st[:, None] + np.arange(n)[None, :])[pick]]] = True


def choose_decision(df, pcol, q_eval, n_true_eval, cfg, allow_exclusive):
    """Try decoders x exclusivity on OOF predictions; return best setting and a results table."""
    results = []
    excl_opts = {"auto": [False, True] if allow_exclusive else [False], "on": [True], "off": [False]}[cfg.exclusivity]
    dec_opts = ["threshold", "expected_f"] if cfg.decoder == "auto" else [cfg.decoder]
    y = df["label"].values.astype(bool)
    for ex in excl_opts:
        keep = exclusive_mask(df, pcol) if ex else np.ones(len(df), bool)
        q, p, yy = df["q"].values[keep], df[pcol].values[keep], y[keep]
        if "threshold" in dec_opts:
            best = (-1, None)
            for t in np.round(np.arange(0.05, 0.99, 0.01), 2):
                s = p >= t
                f, _ = macro_f(q_eval, n_true_eval, q[s], yy[s], cfg.beta)
                if f > best[0]:
                    best = (f, t)
            results.append({"exclusive": ex, "decoder": "threshold", "param": float(best[1]), "macro_f05": best[0]})
        if "expected_f" in dec_opts:
            s = expected_f_select(q, p, cfg.beta)
            f, _ = macro_f(q_eval, n_true_eval, q[s], yy[s], cfg.beta)
            results.append({"exclusive": ex, "decoder": "expected_f", "param": None, "macro_f05": f})
    table = pd.DataFrame(results).sort_values("macro_f05", ascending=False)
    return table.iloc[0].to_dict(), table


def apply_decision(df, pcol, choice, beta=0.5):
    keep = exclusive_mask(df, pcol) if choice["exclusive"] else np.ones(len(df), bool)
    sel = np.zeros(len(df), bool)
    idx = np.flatnonzero(keep)
    if choice["decoder"] == "threshold":
        sel[idx] = df[pcol].values[idx] >= choice["param"]
    else:
        sel[idx] = expected_f_select(df["q"].values[idx], df[pcol].values[idx], beta)
    return sel
