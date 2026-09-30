"""Export the matcher's candidate pairs so a GPU machine can score them with the cross-encoder.

Run on the machine that ran `prune` (reads only cached work/ files, ~2 min):
    python -m src.export_pairs --work work --out ce_export

Writes ce_export/pairs_train.parquet, pairs_test.parquet (q, c record positions) and align.json
(record counts + sampled ids, so the GPU machine can verify it reads the records in the same order).
Zip the folder and upload it; bring back ce_train.parquet / ce_test.parquet into work/.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="ce_export")
    ap.add_argument("--samples", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    align = {}
    for split in ("train", "test"):
        pr = pd.read_parquet(os.path.join(a.work, f"pr_{split}.parquet"), columns=["q", "c"])
        pr.astype({"q": np.int32, "c": np.int32}).to_parquet(os.path.join(a.out, f"pairs_{split}.parquet"),
                                                             compression="zstd")
        ids = pd.read_parquet(os.path.join(a.work, f"rec_{split}.parquet"), columns=["id"])["id"].to_numpy()
        pos = np.unique(np.linspace(0, len(ids) - 1, a.samples).astype(np.int64))
        align[split] = {"n": int(len(ids)), "sample": {str(int(g)): str(ids[g]) for g in pos}}
        print(f"{split}: {len(pr):,} pairs, {len(ids):,} records")
    json.dump(align, open(os.path.join(a.out, "align.json"), "w"))
    print(f"done -> {a.out} (zip this folder and upload it)")


if __name__ == "__main__":
    main()
