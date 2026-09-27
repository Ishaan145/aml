"""Exact macro F0.5 as defined in the Amazon ML Challenge 2026 README.

Per Source-1 entity:
  * true list empty  -> 1.0 if predicted list empty, else 0.0   (singleton rule)
  * true list non-empty, prediction empty -> 0.0
  * otherwise F_beta = (1+b^2) P R / (b^2 P + R)
Macro average over every S1 entity in the evaluation set (the ground-truth keys).
"""
import csv
import sys

csv.field_size_limit(sys.maxsize)


def f_beta_entity(pred, true, beta=0.5):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred, gt, beta=0.5):
    """pred, gt: dict S1_id -> set of S2/S3 ids. S1s missing from pred count as empty."""
    if not gt:
        raise ValueError("empty ground truth")
    return sum(f_beta_entity(pred.get(k, set()), v, beta) for k, v in gt.items()) / len(gt)


def breakdown(pred, gt, beta=0.5):
    """Score split into singletons vs non-singletons, plus how much each group costs you."""
    sing = [k for k, v in gt.items() if not v]
    non = [k for k, v in gt.items() if v]
    fs = [f_beta_entity(pred.get(k, set()), gt[k], beta) for k in sing]
    fn = [f_beta_entity(pred.get(k, set()), gt[k], beta) for k in non]
    n = len(gt)
    return {
        "macro_f05": (sum(fs) + sum(fn)) / n,
        "n_singletons": len(sing),
        "singleton_acc": sum(fs) / max(len(fs), 1),
        "non_singleton_f05": sum(fn) / max(len(fn), 1),
        "points_lost_on_singletons": (len(fs) - sum(fs)) / n,
        "points_lost_on_non_singletons": (len(fn) - sum(fn)) / n,
    }


def read_id_lists(path):
    """Read a 2-column TSV (header + S1 id + comma list) into dict S1 -> set."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if not s1:
                continue
            rest = rest.strip()
            out[s1] = set(rest.split(",")) if rest else set()
    return out


if __name__ == "__main__":
    # Self-test: README worked example must give 0.714
    ex = f_beta_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(ex - 0.7142857) < 1e-6, ex
    assert f_beta_entity(set(), set()) == 1.0 and f_beta_entity({"S2-1"}, set()) == 0.0
    print(f"self-test OK: README example = {ex:.3f}")
    if len(sys.argv) == 3:  # python metric.py predictions.tsv ground_truth.tsv
        print(breakdown(read_id_lists(sys.argv[1]), read_id_lists(sys.argv[2])))
