"""
Shared helpers: reproducibility, signal filtering, AAMI metrics, plotting.
Imported by the numbered pipeline scripts.
"""
import json
import random
import numpy as np

try:
    from scipy.signal import butter, filtfilt
except Exception:  # scipy is required for the real pipeline; synthetic smoke-test still works
    butter = filtfilt = None

import config as C


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------
def bandpass(sig: np.ndarray, fs: int = C.FS,
             low: float = C.BP_LOW, high: float = C.BP_HIGH,
             order: int = C.BP_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    if butter is None:
        raise RuntimeError("scipy is required for filtering (pip install scipy)")
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def zscore_beat(x: np.ndarray) -> np.ndarray:
    """Per-beat z-score normalisation (row-wise for a 2-D array)."""
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - mu) / sd


# --------------------------------------------------------------------------
# AAMI evaluation metrics
# --------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, n_classes: int = len(C.CLASSES)) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def aami_metrics(y_true, y_pred) -> dict:
    """
    Per-class sensitivity (recall), positive predictive value (precision) and F1,
    plus overall accuracy and macro-F1. Returns a JSON-serialisable dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(C.CLASSES)
    cm = confusion_matrix(y_true, y_pred, n)
    per_class = {}
    f1s = []
    for i, name in enumerate(C.CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        se = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity / recall
        ppv = tp / (tp + fp) if (tp + fp) else 0.0         # precision
        f1 = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0
        support = int(cm[i, :].sum())
        per_class[name] = {"Se": round(se, 4), "PPV": round(ppv, 4),
                            "F1": round(f1, 4), "support": support}
        if support > 0:
            f1s.append(f1)
    overall_acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"overall_accuracy": round(overall_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist()}


def print_metrics(title: str, m: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  Overall accuracy : {m['overall_accuracy']:.4f}")
    print(f"  Macro-F1         : {m['macro_f1']:.4f}")
    print(f"  {'class':<6}{'Se':>8}{'PPV':>8}{'F1':>8}{'support':>10}")
    for name, d in m["per_class"].items():
        print(f"  {name:<6}{d['Se']:>8.3f}{d['PPV']:>8.3f}{d['F1']:>8.3f}{d['support']:>10}")


def save_json(obj: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# Plotting (matplotlib; no seaborn dependency)
# --------------------------------------------------------------------------
def plot_confusion(cm, title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(C.CLASSES)):
        for j in range(len(C.CLASSES)):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
