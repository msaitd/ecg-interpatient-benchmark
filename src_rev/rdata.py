"""
Data loading for the revision experiments.

Reads the beats produced by src/02_preprocess.py (data/processed/train.npz = DS1,
test.npz = DS2), computes RR-interval features on the full beat sequence of each record,
then removes the Q class (15 beats) and returns four-class (N, S, V, F) arrays.
"""
import numpy as np
import rconfig as C


def rr_features(rr, rec, window=None):
    """Five RR features per beat, computed record by record in temporal order.

    pre-RR, post-RR, causal local-average RR (mean of the current and the preceding
    window-1 pre-RR values), and the two ratios pre/local and post/local.
    Missing intervals at record borders (stored as 0) are replaced by the other interval.
    """
    window = window or C.RR_LOCAL_WINDOW
    pre = rr[:, 0].astype(np.float64).copy()
    post = rr[:, 1].astype(np.float64).copy()
    feats = np.zeros((len(rr), 5), dtype=np.float64)
    starts = np.r_[0, np.where(rec[1:] != rec[:-1])[0] + 1, len(rec)]
    for a, b in zip(starts[:-1], starts[1:]):
        p, q = pre[a:b], post[a:b]
        med = np.median(p[p > 0]) if np.any(p > 0) else 0.8
        p = np.where(p > 0, p, np.where(q > 0, q, med))
        q = np.where(q > 0, q, p)
        cs = np.concatenate([[0.0], np.cumsum(p)])
        idx = np.arange(len(p))
        lo = np.maximum(0, idx - window + 1)
        local = (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)
        feats[a:b] = np.stack([p, q, local, p / local, q / local], axis=1)
    return feats


def _rf_features(Xraw, rr):
    """The 28 interpretable features of the original random-forest reference."""
    from scipy.stats import skew, kurtosis
    Xz = (Xraw - Xraw.mean(1, keepdims=True)) / (Xraw.std(1, keepdims=True) + 1e-8)
    stats = np.stack([Xraw.mean(1), Xraw.std(1), Xraw.min(1), Xraw.max(1),
                      Xraw.max(1) - Xraw.min(1), (Xraw ** 2).mean(1), np.abs(Xraw).mean(1),
                      skew(Xz, axis=1), kurtosis(Xz, axis=1)], axis=1)
    idx = np.linspace(0, Xraw.shape[1] - 1, 16).astype(int)
    ratio = (rr[:, 0] / (rr[:, 1] + 1e-6)).reshape(-1, 1)
    return np.concatenate([stats, Xz[:, idx], rr[:, :2], ratio], axis=1).astype(np.float32)


def load_split(name, subsample=None, seed=0):
    d = np.load(C.DATA_PROC / f"{name}.npz")
    Xraw = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    rec = d["rec"].astype(np.int64)
    rr_raw = d["rr"].astype(np.float32)
    rr = rr_features(rr_raw, rec)                 # computed BEFORE removing Q beats
    keep = y < 4                                   # drop Q (AAMI class index 4)
    if subsample:                                  # smoke test only
        rng = np.random.default_rng(seed)
        idx = np.where(keep)[0]
        chosen = []
        for c in range(4):                         # keep every class represented
            ci = idx[y[idx] == c]
            n = max(5, int(subsample * len(ci) / len(idx)))
            chosen.append(rng.choice(ci, size=min(n, len(ci)), replace=False))
        keep = np.zeros_like(keep)
        keep[np.sort(np.concatenate(chosen))] = True
    Xraw, y, rec, rr, rr_raw = Xraw[keep], y[keep], rec[keep], rr[keep], rr_raw[keep]
    X = (Xraw - Xraw.mean(1, keepdims=True)) / (Xraw.std(1, keepdims=True) + 1e-8)
    return dict(X=X[:, None, :].astype(np.float32), y=y, rec=rec,
                rr=rr.astype(np.float32), rf=_rf_features(Xraw, rr_raw))


def load_all(subsample=None):
    return {"train": load_split("train", subsample), "test": load_split("test", subsample)}


def standardize(train_rr, *others):
    """z-score RR features with statistics of the training partition only."""
    mu = train_rr.mean(0, keepdims=True)
    sd = train_rr.std(0, keepdims=True) + 1e-6
    out = [((train_rr - mu) / sd).astype(np.float32)]
    out += [((o - mu) / sd).astype(np.float32) for o in others]
    return out


def class_counts(y):
    return np.bincount(y, minlength=C.K).astype(np.int64)


def inverse_frequency_weights(y):
    """w_c = N / (K * n_c), computed on the training labels (not normalized further)."""
    n = np.maximum(class_counts(y), 1).astype(np.float64)
    return n.sum() / (C.K * n)
