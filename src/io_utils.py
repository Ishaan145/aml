"""Robust TSV input/output.

Reads with plain line splitting instead of a CSV parser: addresses may contain stray quote
characters, and a business may literally be called "NA"/"None" — both break pandas defaults.
"""
import os
import time

import numpy as np
import pandas as pd

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_source(path, src):
    """Return DataFrame[id, src, name, addr, country] with every row kept, all strings."""
    ids, names, addrs, ctrys = [], [], [], []
    bad = 0
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        pos = {c.strip().lower(): i for i, c in enumerate(header)}
        missing = [c for c in SRC_COLS if c not in pos]
        if missing:
            raise ValueError(f"{path}: header {header} lacks {missing}")
        ii, ni, ai, ci = (pos[c] for c in SRC_COLS)
        width = len(header)
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            p = line.split("\t")
            if len(p) != width:
                bad += 1
                if len(p) < width:
                    p = p + [""] * (width - len(p))
                else:  # extra tabs: assume they belong to the address
                    extra = len(p) - width
                    p = p[:ai] + ["\t".join(p[ai:ai + extra + 1])] + p[ai + extra + 1:]
            ids.append(p[ii].strip())
            names.append(p[ni])
            addrs.append(p[ai])
            ctrys.append(p[ci].strip())
    if bad:
        log(f"  {os.path.basename(path)}: {bad} rows with unexpected field count were repaired, not dropped")
    df = pd.DataFrame({"id": ids, "name": names, "addr": addrs, "country": ctrys})
    df["src"] = np.int8(src)
    return df


def load_split(data_dir, split):
    parts = []
    for k in (1, 2, 3):
        path = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
        df = read_source(path, k)
        log(f"  read {split} S{k}: {len(df):,} rows")
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["split"] = split
    return out


def read_ground_truth(path):
    """Return (all S1 ids in GT, DataFrame[s1_id, m_id] of true pairs)."""
    s1_all, a, b = [], [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.rstrip("\r\n").partition("\t")
            s1 = s1.strip()
            if not s1:
                continue
            s1_all.append(s1)
            for m in rest.split(","):
                m = m.strip()
                if m:
                    a.append(s1)
                    b.append(m)
    return s1_all, pd.DataFrame({"s1_id": a, "m_id": b}).drop_duplicates()


def write_id_lists(path, s1_ids, pairs_s1, pairs_m, header_col):
    """Write one row per S1 id (in the given order); empty list when S1 has no pairs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    grouped = {}
    if len(pairs_s1):
        df = pd.DataFrame({"s": pairs_s1, "m": pairs_m}).drop_duplicates()
        grouped = df.groupby("s", sort=False)["m"].apply(lambda x: ",".join(x)).to_dict()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s in s1_ids:
            f.write(f"{s}\t{grouped.get(s, '')}\n")
