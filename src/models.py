"""LightGBM with cross-fitting grouped by S1 entity (no entity appears in both train and valid folds)."""
import lightgbm as lgb
import numpy as np

from .io_utils import log


def lgb_params(cfg):
    return dict(objective="binary", learning_rate=cfg.lgb_lr, num_leaves=cfg.lgb_leaves, min_data_in_leaf=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
                num_threads=cfg.n_jobs, seed=cfg.seed, verbose=-1)


def folds_by_entity(q, n_folds, seed):
    table = np.random.default_rng(seed).integers(0, n_folds, int(q.max()) + 1).astype(np.int8)
    return table[q]


def crossfit(X, y, q, cfg, n_folds=2, name="model"):
    """Out-of-fold predictions for every row + mean best iteration."""
    fold = folds_by_entity(q, n_folds, cfg.seed)
    oof = np.zeros(len(y), dtype=np.float32)
    iters, models = [], []
    for k in range(n_folds):
        tr, va = fold != k, fold == k
        dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(lgb_params(cfg), dtr, num_boost_round=cfg.lgb_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg.lgb_early_stop, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration or cfg.lgb_rounds)
        models.append(m)
        log(f"  {name} fold {k}: best_iter={iters[-1]}, valid rows={va.sum():,}, positives={int(y[va].sum()):,}")
    return oof, int(np.mean(iters) * 1.1) + 1, FoldEnsemble(models)


class FoldEnsemble:
    """Mean of the cross-fitting fold models: used for test inference instead of a full refit
    (saves one full training pass; each fold model already saw half of the entities)."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in self.models], axis=0)

    def feature_importance(self, kind="gain"):
        return np.sum([m.feature_importance(kind) for m in self.models], axis=0)

    def save_model(self, path):
        for i, m in enumerate(self.models):
            m.save_model(f"{path}.fold{i}")


def fit_full(X, y, cfg, rounds):
    return lgb.train(lgb_params(cfg), lgb.Dataset(X, y), num_boost_round=rounds)


def importance(model, names, top=25):
    gain = model.feature_importance("gain")
    order = np.argsort(-gain)[:top]
    tot = gain.sum() or 1.0
    return [(names[i], round(float(gain[i] / tot), 4)) for i in order]
