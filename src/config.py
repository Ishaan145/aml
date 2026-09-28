"""All tunable knobs in one place. Override any field from the CLI: --set name=value."""
from dataclasses import dataclass, fields


@dataclass
class Config:
    # ---- compute ----
    n_jobs: int = 4                  # CPU workers for normalization / rapidfuzz / sparse top-k
    query_chunk: int = 50_000        # S1 queries per blocking chunk (bounds peak memory)
    seed: int = 42

    # ---- normalization ----
    mine_min_count: int = 25         # min co-occurrences in train matches to accept a token equivalence
    mine_max_group: int = 6          # reject equivalence groups larger than this (likely noise)
    generic_top_n: int = 60          # per-country most frequent name tokens treated as "generic"
    generic_df_frac: float = 0.004   # ...plus any name token in more than this share of records

    # ---- blocking (candidate generation) ----
    partition_by_country: bool = True
    tok_topk: int = 20               # top-k per S1 from the word-token IDF view
    chr_topk: int = 0                # top-k per S1 from the char n-gram view (0 = off: on the real data
                                     #   it added 0.25% unique recall for ~40% of blocking time)
    rev_topk: int = 5                # reverse view: top-k S1s per S2/S3 record (0 = off)
    rev_keep_rank: int = 2           # reverse hits with rank < this always survive the union cap
    chr_ngram: int = 4               # char n-gram size for the typo-robust name view
    tok_max_df: int = 20_000         # block purging cap: drop tokens in more pool records than this
    chr_max_df: int = 20_000
    tok_budget: int = 40_000         # max mean postings scanned per query (bounds blocking time);
    chr_budget: int = 40_000         #   the purge cut-off is lowered automatically to fit it
                                     # chr_topk=0 switches the char n-gram view off entirely
    min_sim: float = 0.05            # drop retrieved pairs below this cosine
    key_max_block: int = 50          # exact-key blocks larger than this are skipped
    key_keep_block: int = 10         # hits from blocks this small bypass the union cap (very specific keys)
    union_cap: int = 40              # keep at most this many union candidates per S1 (by RRF)
    w_name: float = 1.0              # field weights inside the token view
    w_addr: float = 0.6
    w_postal: float = 1.2

    # ---- pruner (last filtering stage -> candidate_pairs.tsv) ----
    prune_recall_loss: float = 0.002  # allowed share of blocked true pairs lost by pruning
    prune_max: int = 20               # hard cap on candidates per S1 after pruning

    # ---- matcher ----
    lgb_rounds: int = 1000
    lgb_lr: float = 0.1
    lgb_leaves: int = 127
    lgb_early_stop: int = 50

    # ---- decision layer ----
    beta: float = 0.5
    decoder: str = "auto"            # auto | threshold | expected_f
    exclusivity: str = "auto"        # auto | on | off

    @classmethod
    def from_overrides(cls, overrides):
        cfg = cls()
        types = {f.name: f.type for f in fields(cls)}
        for item in overrides or []:
            k, v = item.split("=", 1)
            if k not in types:
                raise KeyError(f"unknown config key {k}")
            t = types[k]
            if t in (bool, "bool"):
                v = v.lower() in ("1", "true", "yes", "on")
            elif t in (int, "int"):
                v = int(v)
            elif t in (float, "float"):
                v = float(v)
            setattr(cfg, k, v)
        return cfg
