"""Text normalization, postcode extraction, mined token equivalences, generic-token detection.

Nothing here looks anything up externally. The abbreviation table below encodes only the
patterns the README lists; everything else (token equivalences, generic tokens) is learned
from the provided data.
"""
import re
import sys
import unicodedata
from collections import Counter
from multiprocessing import Pool

import numpy as np
from anyascii import anyascii

# Strip all combining marks (é -> e; Indic vowel signs) so words never split; base letters of every script stay.
_COMBINING = {cp: None for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)).startswith("M")}
_APOS = re.compile(r"['’`´]")
_DOTCOM = re.compile(r"\.(com|net|org)\b", re.IGNORECASE)     # apexallied.com -> apexallied com
_NONWORD = re.compile(r"[\W_]+", re.UNICODE)
_PIN_SPACED = re.compile(r"\b(\d{3})\s(\d{3})\b")
_POSTAL = re.compile(r"\b\d{5,6}\b")
_DIG_ALPHA = re.compile(r"(\d)([^\W\d_])")
_ALPHA_DIG = re.compile(r"([^\W\d_])(\d)")

# Generic English abbreviations named in the README noise list. Applied to both sides of every
# comparison, so a mapping only has to be consistent, not linguistically perfect.
ABBREV = {
    # names
    "corp": "corporation", "corpn": "corporation", "inc": "incorporated", "incorp": "incorporated",
    "co": "company", "ltd": "limited", "pvt": "private", "bros": "brothers", "mfg": "manufacturing",
    "intl": "international", "svcs": "services", "svc": "services", "and": "and",
    # addresses
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "boul": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite", "apt": "apartment",
    "fl": "floor", "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite", "mkt": "market",
    "ngr": "nagar", "clny": "colony", "sec": "sector", "mn": "main", "crs": "cross", "stn": "station",
    "hosp": "hospital", "govt": "government", "no": "number",
}


def _has_non_latin(s):
    return not s.isascii() and any(ord(ch) > 0x24F for ch in s)


def basic_norm(s):
    s = _DOTCOM.sub(r" \1", s)
    if _has_non_latin(s):                    # Devanagari/Tamil/Bengali/... -> Latin letters (anyascii, ISC)
        s = anyascii(s)                      #   so Indic-script names can match their Latin-script twins
    s = unicodedata.normalize("NFKD", s).translate(_COMBINING).casefold()
    s = s.replace(".", "")                   # L.L.C. -> llc, Pvt. -> pvt
    s = s.replace("&", " and ")
    s = _APOS.sub("", s)                     # domino's -> dominos
    s = _NONWORD.sub(" ", s)
    s = _ALPHA_DIG.sub(r"\1 \2", _DIG_ALPHA.sub(r"\1 \2", s))   # 4430sunset -> 4430 sunset
    return " ".join(ABBREV.get(t, t) for t in s.split())


def extract_postal(raw_addr):
    s = _PIN_SPACED.sub(r"\1\2", raw_addr)
    m = _POSTAL.findall(s)
    return m[-1] if m else ""


def _norm_chunk(args):
    names, addrs = args
    return ([basic_norm(x) for x in names], [basic_norm(x) for x in addrs],
            [extract_postal(x) for x in addrs])


def _map_chunk(args):
    texts, mapping = args
    return [" ".join(mapping.get(t, t) for t in s.split()) for s in texts]


def _chunks(n, size):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def normalize_frame(df, n_jobs, chunk=200_000):
    """Adds name_n, addr_n, postal columns (basic normalization + README abbreviations)."""
    names, addrs = df["name"].tolist(), df["addr"].tolist()
    jobs = [(names[a:b], addrs[a:b]) for a, b in _chunks(len(df), chunk)]
    if n_jobs > 1 and len(jobs) > 1:
        with Pool(n_jobs) as p:
            res = p.map(_norm_chunk, jobs)
    else:
        res = [_norm_chunk(j) for j in jobs]
    df["name_n"] = [x for r in res for x in r[0]]
    df["addr_n"] = [x for r in res for x in r[1]]
    df["postal"] = [x for r in res for x in r[2]]
    return df


def apply_mapping(df, mapping, n_jobs, chunk=300_000):
    if not mapping:
        return df
    for col in ("name_n", "addr_n"):
        texts = df[col].tolist()
        jobs = [(texts[a:b], mapping) for a, b in _chunks(len(texts), chunk)]
        if n_jobs > 1 and len(jobs) > 1:
            with Pool(n_jobs) as p:
                res = p.map(_map_chunk, jobs)
        else:
            res = [_map_chunk(j) for j in jobs]
        df[col] = [x for r in res for x in r]
    return df


def mine_equivalences(text_pairs, min_count=25, max_group=6):
    """Learn token equivalences (rd<->road, bangalore<->bengaluru, ...) from matched pairs.

    A pair of records that differs by exactly one token on each side is evidence that those
    two tokens are interchangeable. Pairs seen >= min_count times are merged with union-find;
    each group maps to its most frequent member. Tokens containing digits are never merged
    (house numbers and postcodes must stay distinct).
    """
    sub, freq = Counter(), Counter()
    for x, y in text_pairs:
        a, b = set(x.split()), set(y.split())
        freq.update(a)
        freq.update(b)
        d1, d2 = a - b, b - a
        if len(d1) == 1 and len(d2) == 1:
            t1, t2 = next(iter(d1)), next(iter(d2))
            if any(c.isdigit() for c in t1 + t2):
                continue
            sub[(t1, t2) if t1 < t2 else (t2, t1)] += 1
    parent = {}

    def find(t):
        parent.setdefault(t, t)
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    kept = [(p, c) for p, c in sub.items()
            if c >= min_count and c >= 0.01 * min(freq[p[0]], freq[p[1]])]
    for (t1, t2), _ in kept:
        r1, r2 = find(t1), find(t2)
        if r1 != r2:
            parent[r1] = r2
    groups = {}
    for t in parent:
        groups.setdefault(find(t), []).append(t)
    mapping = {}
    for members in groups.values():
        if len(members) < 2 or len(members) > max_group:
            continue
        canon = max(members, key=lambda t: (freq[t], -len(t)))
        for t in members:
            if t != canon:
                mapping[t] = canon
    return mapping, sorted(kept, key=lambda x: -x[1])


def add_name_core(df, top_n, df_frac):
    """name_core = name tokens minus this country's most generic tokens (learned per country,
    so it also works for an unseen country such as France). Falls back to the full name."""
    core = np.empty(len(df), dtype=object)
    generic_by_country = {}
    for c, idx in df.groupby("country", sort=False).indices.items():
        names = df["name_n"].values[idx]
        cnt = Counter(t for s in names for t in set(s.split()))
        n = max(len(names), 1)
        generic = {t for t, _ in cnt.most_common(top_n)} | {t for t, v in cnt.items() if v / n > df_frac}
        generic_by_country[c] = generic
        out = []
        for s in names:
            toks = [t for t in s.split() if t not in generic]
            out.append(" ".join(toks) if toks else s)
        core[idx] = out
    df["name_core"] = core
    return df, generic_by_country
