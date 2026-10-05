"""
Metrics (NumPy only, no deep-learning dependency).

Definitions used throughout the revised manuscript:
  sensitivity (Se) = TP / (TP + FN); positive predictive value (PPV) = TP / (TP + FP)
  F1 = 2 Se PPV / (Se + PPV). When a class receives no positive prediction (TP + FP = 0)
  its PPV is set to 0, and F1 = 0 whenever Se + PPV = 0 (zero-division convention).
  macro-F1 = unweighted mean of the per-class F1 over the classes that have at least one
  true beat in the evaluated set (all four classes N, S, V, F in DS2 and in the pooled
  cross-validation predictions).
"""
import numpy as np

K = 4


def confusion(y, p, k=K):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(p, int)), 1)
    return cm


def per_class(cm):
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pp = cm.sum(0).astype(float)
    se = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    ppv = np.divide(tp, pp, out=np.zeros_like(tp), where=pp > 0)
    f1 = np.divide(2 * se * ppv, se + ppv, out=np.zeros_like(tp), where=(se + ppv) > 0)
    return se, ppv, f1, sup


def summarize(y, p, k=K):
    cm = confusion(y, p, k)
    se, ppv, f1, sup = per_class(cm)
    present = sup > 0
    return dict(acc=float(np.trace(cm) / max(cm.sum(), 1)),
                macro_f1=float(f1[present].mean()) if present.any() else 0.0,
                se=se.tolist(), ppv=ppv.tolist(), f1=f1.tolist(),
                support=sup.astype(int).tolist(), cm=cm.tolist())


def macro_f1_from_cm(cm):
    se, ppv, f1, sup = per_class(cm)
    present = sup > 0
    return float(f1[present].mean()) if present.any() else 0.0
