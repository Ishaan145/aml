"""Transformer cross-encoder re-scorer (runs on a GPU, e.g. Kaggle T4 x2).

Reads both records of a candidate pair together ("name | address" [SEP] "name | address") and
outputs a match logit. Its score, and its rank/gap among the other candidates of the same S1 (the
"selective" / listwise idea of Mistral4SelectEM, AEI 2025), become extra features of the LightGBM
matcher (pipeline stage `match` picks up work/ce_train.parquet and work/ce_test.parquet).

Default model: intfloat/multilingual-e5-small (MIT, 118M params, 100+ languages incl. Indic scripts
and French). Licence and size are within the challenge rule (MIT/Apache 2.0, <= 8B).

Leakage control: the cross-encoder is trained ONLY on train S1 entities that the matcher does not use
(the complement of the matcher's --train-frac sample), so its scores on matcher pairs are out-of-sample.

Commands (all need --data-dir; `mine` also needs the prep outputs in --work):
  mine   CPU  block a sample of non-matcher train S1s (forward search + keys), keep top-K per S1
              -> <work>/ce_mine_pairs.parquet (q, c, label)
  train  GPU  fine-tune on the mined pairs -> <model-out>/ (HF format)
  score  GPU  score a (q, c) pairs parquet of one split -> parquet (q, c, ce)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from .io_utils import load_split, log

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # DataLoader workers tokenize in parallel


# ------------------------------------------------------------------------------------------ text
def record_texts(data_dir, split):
    """Raw 'name | address' per record, in exactly the same order (gidx) as the pipeline's rec files."""
    df = load_split(data_dir, split)
    txt = (df["name"].str.strip() + " | " + df["addr"].str.strip()).to_numpy(dtype=object)
    return txt, df["id"].to_numpy(dtype=object)


def check_alignment(ids, align):
    """align: {"n": total records, "sample": {gidx: id}} written by export_pairs on the other machine."""
    if align is None:
        return
    if int(align["n"]) != len(ids):
        raise RuntimeError(f"record count differs: here {len(ids)}, exporting machine {align['n']}")
    bad = [(int(g), i) for g, i in align["sample"].items() if ids[int(g)] != i]
    if bad:
        raise RuntimeError(f"record order differs from the exporting machine, e.g. {bad[:3]}")
    log(f"  alignment check passed ({len(align['sample'])} sampled records)")


# ------------------------------------------------------------------------------------------ mine
def mine(a):
    from .blocking import generate_candidates
    from .config import Config
    cfg = Config.from_overrides(["chr_topk=0", "rev_topk=0", f"n_jobs={a.n_jobs}"])
    rec = pd.read_parquet(os.path.join(a.work, "rec_train.parquet"))
    for col in ("id", "country", "name_n", "addr_n", "name_core", "postal"):
        rec[col] = rec[col].astype(object)
    rec["gidx"] = np.arange(len(rec), dtype=np.int32)
    n = len(rec)
    # the matcher's S1 sample, reproduced exactly as pipeline.stage_block draws it
    matcher_s1 = np.random.default_rng(a.seed).random(n) < a.matcher_train_frac
    pool_s1 = np.flatnonzero((rec["src"].values == 1) & ~matcher_s1)
    rng = np.random.default_rng(a.seed + 7)
    pick = rng.choice(pool_s1, min(a.n_s1, len(pool_s1)), replace=False)
    mask = np.zeros(n, bool)
    mask[pick] = True
    log(f"mining: {len(pick):,} train S1s outside the matcher sample (forward search + keys)")
    cand = generate_candidates(rec, cfg, mask)
    cand = cand.sort_values(["q", "rrf"], ascending=[True, False])
    cand = cand[cand.groupby("q").cumcount() < a.topk]
    gt = pd.read_parquet(os.path.join(a.work, "gt_train.parquet"))
    keys = set((gt["q"].values.astype(np.int64) * n + gt["c"].values.astype(np.int64)).tolist())
    ck = cand["q"].values.astype(np.int64) * n + cand["c"].values.astype(np.int64)
    cand["label"] = np.fromiter((k in keys for k in ck), np.int8, len(ck))
    out = cand[["q", "c", "label"]].reset_index(drop=True)
    out.to_parquet(os.path.join(a.work, "ce_mine_pairs.parquet"))
    log(f"  mined {len(out):,} pairs, positives {out['label'].mean():.1%}, S1s {out['q'].nunique():,}")


# ------------------------------------------------------------------------------------------ torch
def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Pairs:
    def __init__(self, txt, q, c, y=None):
        self.txt, self.q, self.c, self.y = txt, q, c, y

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return i


def _collate(tok, ds, max_len):
    import torch

    def fn(idx):
        a = [ds.txt[ds.q[i]] for i in idx]
        b = [ds.txt[ds.c[i]] for i in idx]
        enc = tok(a, b, truncation="longest_first", max_length=max_len, padding=True, return_tensors="pt")
        if ds.y is not None:
            enc["labels"] = torch.tensor([float(ds.y[i]) for i in idx])
        enc["idx"] = torch.tensor(idx)
        return enc
    return fn


_ACCEPTED = set()


def _load(model_name, device):
    import inspect
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    # only pass inputs the model accepts (e.g. no token_type_ids for ModernBERT / 1-type BERTs)
    _ACCEPTED.clear()
    _ACCEPTED.update(inspect.signature(model.forward).parameters)
    if getattr(model.config, "type_vocab_size", 2) < 2:
        _ACCEPTED.discard("token_type_ids")
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return tok, model


def _forward(model, batch, device):
    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in _ACCEPTED}
    return model(**inputs).logits.squeeze(-1)


def train(a):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(a.seed)
    device = _device()
    txt, _ = record_texts(a.data_dir, "train")
    pairs = pd.read_parquet(a.pairs)
    if len(pairs) > a.max_pairs:
        pairs = pairs.sample(a.max_pairs, random_state=a.seed)
    pairs = pairs.sample(frac=1.0, random_state=a.seed + 1).reset_index(drop=True)
    log(f"training on {len(pairs):,} pairs ({pairs['label'].mean():.1%} positive), device={device}, "
        f"gpus={torch.cuda.device_count()}, model={a.model}")
    tok, model = _load(a.model, device)
    ds = _Pairs(txt, pairs["q"].values, pairs["c"].values, pairs["label"].values)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=_collate(tok, ds, a.max_len), drop_last=True, pin_memory=device.type == "cuda")
    steps = len(dl) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step, run = time.time(), 0, 0.0
    for ep in range(a.epochs):
        for batch in dl:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = _forward(model, batch, device)
                loss = lossf(logits.float(), batch["labels"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % a.log_every == 0 or step == steps:
                el = time.time() - t0
                log(f"  epoch {ep + 1} step {step}/{steps} loss {run:.4f} | {el:.0f}s, "
                    f"ETA {el / step * (steps - step):.0f}s")
    os.makedirs(a.model_out, exist_ok=True)
    (model.module if hasattr(model, "module") else model).save_pretrained(a.model_out)
    tok.save_pretrained(a.model_out)
    json.dump({"base_model": a.model, "pairs": len(pairs), "epochs": a.epochs, "max_len": a.max_len},
              open(os.path.join(a.model_out, "ce_info.json"), "w"))
    log(f"saved model to {a.model_out} ({time.time() - t0:.0f}s)")


def _merge_shards(shard_dir, n_shards, out):
    paths = [os.path.join(shard_dir, f"{s:04d}.parquet") for s in range(n_shards)]
    if not all(os.path.exists(p) for p in paths):
        return False
    res = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    tmp = out + ".tmp"
    res.to_parquet(tmp)
    os.replace(tmp, out)
    log(f"wrote {out} ({len(res):,} rows)")
    return True


def score(a):
    """With --parts N, run N processes (one per GPU, CUDA_VISIBLE_DEVICES=i, --part i): each scores every
    N-th shard; a final call with --parts 1 (or any part once all shards exist) merges them."""
    import torch
    from torch.utils.data import DataLoader
    pairs = pd.read_parquet(a.pairs, columns=["q", "c"])
    shard_dir = a.shard_dir or a.out + ".shards"
    os.makedirs(shard_dir, exist_ok=True)
    n_shards = math.ceil(len(pairs) / a.shard)
    if a.parts == 1 and _merge_shards(shard_dir, n_shards, a.out):
        return                                         # everything already scored: merge only
    mine = [s for s in range(n_shards) if s % a.parts == a.part
            and not os.path.exists(os.path.join(shard_dir, f"{s:04d}.parquet"))]
    if not mine:
        log(f"part {a.part}/{a.parts}: nothing left to score")
        return
    device = _device()
    txt, ids = record_texts(a.data_dir, a.split)
    align = None
    if a.align and os.path.exists(a.align):
        align = json.load(open(a.align)).get(a.split)
    check_alignment(ids, align)
    info_path = os.path.join(a.model_dir, "ce_info.json")
    max_len = json.load(open(info_path))["max_len"] if os.path.exists(info_path) else a.max_len
    tok, model = _load(a.model_dir, device)
    model.eval()
    # sort by text length so each batch pads to a similar length (much faster)
    lens = np.fromiter((len(txt[q]) + len(txt[c]) for q, c in zip(pairs["q"].values, pairs["c"].values)),
                       np.int32, len(pairs))
    order = np.argsort(lens, kind="stable")
    q_all, c_all = pairs["q"].values[order], pairs["c"].values[order]
    log(f"part {a.part}/{a.parts}: scoring {len(mine)} of {n_shards} shards ({len(pairs):,} {a.split} pairs "
        f"in total), device={device}, gpus={torch.cuda.device_count()}")
    t0 = time.time()
    use_amp = device.type == "cuda"
    for done, s in enumerate(mine, 1):
        path = os.path.join(shard_dir, f"{s:04d}.parquet")
        lo, hi = s * a.shard, min((s + 1) * a.shard, len(pairs))
        ds = _Pairs(txt, q_all[lo:hi], c_all[lo:hi])
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                        collate_fn=_collate(tok, ds, max_len), pin_memory=device.type == "cuda")
        out = np.empty(hi - lo, np.float32)
        with torch.inference_mode():
            for batch in dl:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = _forward(model, batch, device)
                out[batch["idx"].numpy()] = logits.float().cpu().numpy()
        pd.DataFrame({"q": q_all[lo:hi], "c": c_all[lo:hi], "ce": out}).to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        el = time.time() - t0
        log(f"  part {a.part}: shard {done}/{len(mine)} | {el:.0f}s, {(done * a.shard) / el:.0f} pairs/s, "
            f"ETA {el / done * (len(mine) - done):.0f}s")
    if a.parts == 1:
        _merge_shards(shard_dir, n_shards, a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mine", "train", "score"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    # mine
    ap.add_argument("--matcher-train-frac", type=float, default=0.3,
                    help="the --train-frac of the main pipeline run (its S1s are excluded here)")
    ap.add_argument("--n-s1", type=int, default=200_000)
    ap.add_argument("--topk", type=int, default=8)
    # train
    ap.add_argument("--pairs", help="parquet with q, c (and label for train)")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--model-out", default="ce_model")
    ap.add_argument("--max-pairs", type=int, default=1_500_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=200)
    # score
    ap.add_argument("--model-dir", default="ce_model")
    ap.add_argument("--split", choices=["train", "test"])
    ap.add_argument("--align", help="align.json written by export_pairs (record-order check)")
    ap.add_argument("--shard", type=int, default=250_000)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--shard-dir", help="where partial results go (resume point); default <out>.shards")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"mine": mine, "train": train, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
