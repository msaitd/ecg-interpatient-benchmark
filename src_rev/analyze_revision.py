"""
Statistical analysis of the revision runs (NumPy / SciPy / matplotlib only; no torch needed).

  python src_rev/analyze_revision.py                 # analyses results_rev/
  python src_rev/analyze_revision.py --root PATH --B 2000

Uncertainty
  * mean +/- SD over the five seeds
  * 95% confidence intervals from a record-level (patient-level) bootstrap: records of the
    evaluated set are resampled with replacement (B replicates); for each replicate the
    metric is computed for every seed and averaged over seeds; percentile intervals.
  * paired differences (e.g. beat-wise minus patient-wise, or a configuration minus the
    primary model) use the SAME resampled records for both arms; two-sided bootstrap p value
    p = 2 min(P(delta* <= 0), P(delta* >= 0)), bounded below by 1/B.
  * paired t-test across seeds as a secondary test for the protocol effect.
Outputs: <root>/analysis/summary.json, tables/*.csv, figures/*.png (300 dpi)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rconfig as C  # noqa: E402

K = C.K
CLS = C.CLASSES
LABEL = {
    "cnn_ce": "1D-CNN, cross-entropy (primary)",
    "cnn_cw": "1D-CNN, class-weighted cross-entropy",
    "cnn_os": "1D-CNN, class-balanced oversampling",
    "cnnrr_ce": "1D-CNN + RR-interval features",
    "resnet_ce": "1D-ResNet (deeper comparator)",
    "rf": "Random forest (28 features incl. RR)",
}
for g in (0.5, 1, 2, 3, 5):
    LABEL[f"cnn_focal_g{g:g}"] = f"1D-CNN, focal loss (gamma={g:g})"
DS_ORDER = ["cnn_ce", "cnn_cw", "cnn_os"] + [f"cnn_focal_g{g:g}" for g in (0.5, 1, 2, 3, 5)] + \
           ["cnnrr_ce", "resnet_ce", "rf"]


# ------------------------------------------------------------------ helpers
def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics_vec(cm):
    """cm (..., K, K) -> dict of arrays over the leading dimensions."""
    cm = np.asarray(cm, float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup = cm.sum(-1)
    pp = cm.sum(-2)
    se = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    ppv = np.divide(tp, pp, out=np.zeros_like(tp), where=pp > 0)
    f1 = np.divide(2 * se * ppv, se + ppv, out=np.zeros_like(tp), where=(se + ppv) > 0)
    present = sup > 0
    macro = (f1 * present).sum(-1) / np.maximum(present.sum(-1), 1)
    acc = tp.sum(-1) / np.maximum(cm.sum((-1, -2)), 1)
    return dict(acc=acc, macro_f1=macro, se=se, ppv=ppv, f1=f1, support=sup)


def record_cms(y, pred, rec, records):
    out = np.zeros((len(records), K, K), np.int64)
    for i, r in enumerate(records):
        m = rec == r
        if m.any():
            np.add.at(out[i], (y[m], pred[m]), 1)
    return out


class Boot:
    def __init__(self, n_records, B, seed=2026):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_records, size=(B, n_records))
        self.W = np.stack([np.bincount(r, minlength=n_records) for r in idx]).astype(float)

    def cms(self, rcms):                      # rcms (S, R, K, K) -> (B, S, K, K)
        return np.einsum("br,srkl->bskl", self.W, rcms)


def ci(a):
    a = np.asarray(a, float)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def pval(delta):
    delta = np.asarray(delta, float)
    p = 2 * min((delta <= 0).mean(), (delta >= 0).mean())
    return float(min(max(p, 1.0 / len(delta)), 1.0))


def msd(a):
    a = np.asarray(a, float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_preds(d, stem):
    files = sorted(d.glob(f"{stem}_s*.npz"), key=lambda p: int(p.stem.rsplit("_s", 1)[1]))
    files = [f for f in files if f.stem.rsplit("_s", 1)[0] == stem]
    preds, folds = [], []
    for f in files:
        z = np.load(f)
        preds.append(z["probs"].astype(np.float32).argmax(1))
        folds.append(z["fold"] if "fold" in z.files else None)
    return preds, folds, [int(f.stem.rsplit("_s", 1)[1]) for f in files]


# ------------------------------------------------------------------ analyses
def analyse_ds(root, B):
    d = root / "ds"
    lab = np.load(d / "labels.npz")
    y, rec = lab["y"], lab["rec"]
    records = sorted(set(rec.tolist()))
    boot = Boot(len(records), B)
    res, bootstore = {}, {}
    for name in DS_ORDER:
        preds, _, seeds = load_preds(d, name)
        if not preds:
            continue
        rcms = np.stack([record_cms(y, p, rec, records) for p in preds])
        ps = metrics_vec(rcms.sum(1))                     # per seed
        bm = metrics_vec(boot.cms(rcms))                  # (B, S)
        bstore = {k: bm[k].mean(1) for k in ("acc", "macro_f1")}
        bstore["se"] = bm["se"].mean(1)
        bstore["ppv"] = bm["ppv"].mean(1)
        bstore["f1"] = bm["f1"].mean(1)
        bootstore[name] = bstore
        r = dict(name=name, label=LABEL.get(name, name), seeds=seeds, n_seeds=len(seeds))
        r["acc_mean"], r["acc_sd"] = msd(ps["acc"])
        r["macro_f1_mean"], r["macro_f1_sd"] = msd(ps["macro_f1"])
        r["acc_ci"] = ci(bstore["acc"])
        r["macro_f1_ci"] = ci(bstore["macro_f1"])
        for key in ("se", "ppv", "f1"):
            r[f"{key}_mean"] = ps[key].mean(0).tolist()
            r[f"{key}_sd"] = (ps[key].std(0, ddof=1) if len(seeds) > 1 else np.zeros(K)).tolist()
            r[f"{key}_ci"] = [ci(bstore[key][:, c]) for c in range(K)]
        r["cm_pooled_over_seeds"] = rcms.sum((0, 1)).tolist()
        r["support"] = rcms[0].sum((0, 2)).tolist()
        res[name] = r
    if "cnn_ce" in bootstore:
        for name, r in res.items():
            if name == "cnn_ce":
                continue
            dlt = bootstore[name]["macro_f1"] - bootstore["cnn_ce"]["macro_f1"]
            r["delta_macro_f1_vs_primary"] = r["macro_f1_mean"] - res["cnn_ce"]["macro_f1_mean"]
            r["delta_macro_f1_vs_primary_ci"] = ci(dlt)
            r["delta_macro_f1_vs_primary_p"] = pval(dlt)
            dS = bootstore[name]["se"][:, 1] - bootstore["cnn_ce"]["se"][:, 1]
            r["delta_S_se_vs_primary"] = r["se_mean"][1] - res["cnn_ce"]["se_mean"][1]
            r["delta_S_se_vs_primary_ci"] = ci(dS)
            r["delta_S_se_vs_primary_p"] = pval(dS)
    return res, dict(records=records, n_beats=int(len(y)), support=np.bincount(y, minlength=K).tolist())


def analyse_cv(root, B):
    from scipy import stats
    d = root / "cv"
    lab = np.load(d / "labels.npz")
    y, rec = lab["y"], lab["rec"]
    records = sorted(set(rec.tolist()))
    boot = Boot(len(records), B)
    cells, bst = {}, {}
    for fam in ("cnn", "cnnrr"):
        for proto in ("patient", "beat"):
            preds, folds, seeds = load_preds(d, f"{fam}_ce_{proto}")
            if not preds:
                continue
            rcms = np.stack([record_cms(y, p, rec, records) for p in preds])
            ps = metrics_vec(rcms.sum(1))
            bm = metrics_vec(boot.cms(rcms))
            store = {k: bm[k].mean(1) for k in ("acc", "macro_f1")}
            for k in ("se", "ppv", "f1"):
                store[k] = bm[k].mean(1)
            bst[(fam, proto)] = store
            fold_macro, fold_acc = [], []
            for p, f in zip(preds, folds):
                for fi in sorted(set(f.tolist())):
                    m = f == fi
                    cm = np.zeros((K, K), np.int64)
                    np.add.at(cm, (y[m], p[m]), 1)
                    mv = metrics_vec(cm)
                    fold_macro.append(float(mv["macro_f1"]))
                    fold_acc.append(float(mv["acc"]))
            c = dict(family=fam, protocol=proto, seeds=seeds, n_seeds=len(seeds),
                     seed_macro_f1=ps["macro_f1"].tolist())
            c["acc_mean"], c["acc_sd"] = msd(ps["acc"])
            c["macro_f1_mean"], c["macro_f1_sd"] = msd(ps["macro_f1"])
            c["acc_ci"], c["macro_f1_ci"] = ci(store["acc"]), ci(store["macro_f1"])
            c["foldwise_macro_f1_mean"], c["foldwise_macro_f1_sd"] = msd(fold_macro)
            c["foldwise_acc_mean"], c["foldwise_acc_sd"] = msd(fold_acc)
            c["n_folds_total"] = len(fold_macro)
            for k in ("se", "ppv", "f1"):
                c[f"{k}_mean"] = ps[k].mean(0).tolist()
                c[f"{k}_sd"] = (ps[k].std(0, ddof=1) if len(seeds) > 1 else np.zeros(K)).tolist()
                c[f"{k}_ci"] = [ci(store[k][:, j]) for j in range(K)]
            c["cm_pooled_over_seeds"] = rcms.sum((0, 1)).tolist()
            cells[f"{fam}_{proto}"] = c
    effects = {}

    def eff(name, a, b, metric="macro_f1", cls=None):
        if a not in bst or b not in bst:
            return
        if cls is None:
            ba, bb = bst[a][metric], bst[b][metric]
            pa, pb = cells[f"{a[0]}_{a[1]}"][f"{metric}_mean"], cells[f"{b[0]}_{b[1]}"][f"{metric}_mean"]
        else:
            ba, bb = bst[a][metric][:, cls], bst[b][metric][:, cls]
            pa = cells[f"{a[0]}_{a[1]}"][f"{metric}_mean"][cls]
            pb = cells[f"{b[0]}_{b[1]}"][f"{metric}_mean"][cls]
        dlt = ba - bb
        effects[name] = dict(point=float(pa - pb), ci=ci(dlt), p=pval(dlt))

    for metric, cls, tag in (("macro_f1", None, "macro_f1"), ("se", 1, "S_se"), ("f1", 1, "S_f1"),
                             ("se", 3, "F_se"), ("f1", 3, "F_f1"), ("acc", None, "acc")):
        eff(f"protocol_cnn_{tag}", ("cnn", "beat"), ("cnn", "patient"), metric, cls)
        eff(f"protocol_cnnrr_{tag}", ("cnnrr", "beat"), ("cnnrr", "patient"), metric, cls)
        eff(f"model_patient_{tag}", ("cnnrr", "patient"), ("cnn", "patient"), metric, cls)
        eff(f"model_beat_{tag}", ("cnnrr", "beat"), ("cnn", "beat"), metric, cls)
    if all(k in bst for k in (("cnn", "beat"), ("cnn", "patient"), ("cnnrr", "beat"), ("cnnrr", "patient"))):
        dlt = (bst[("cnnrr", "beat")]["macro_f1"] - bst[("cnnrr", "patient")]["macro_f1"]) - \
              (bst[("cnn", "beat")]["macro_f1"] - bst[("cnn", "patient")]["macro_f1"])
        pt = (cells["cnnrr_beat"]["macro_f1_mean"] - cells["cnnrr_patient"]["macro_f1_mean"]) - \
             (cells["cnn_beat"]["macro_f1_mean"] - cells["cnn_patient"]["macro_f1_mean"])
        effects["interaction_macro_f1"] = dict(point=float(pt), ci=ci(dlt), p=pval(dlt))
    for fam in ("cnn", "cnnrr"):
        a, b = cells.get(f"{fam}_beat"), cells.get(f"{fam}_patient")
        if a and b and len(a["seed_macro_f1"]) == len(b["seed_macro_f1"]) and len(a["seed_macro_f1"]) > 1:
            t = stats.ttest_rel(a["seed_macro_f1"], b["seed_macro_f1"])
            effects[f"protocol_{fam}_macro_f1_paired_t"] = dict(t=float(t.statistic), p=float(t.pvalue),
                                                               df=len(a["seed_macro_f1"]) - 1)
    return cells, effects, dict(records=records, n_beats=int(len(y)), support=np.bincount(y, minlength=K).tolist())


def analyse_tuning(root):
    rows, sel = [], {}
    for fam in ("cnn", "cnnrr", "resnet"):
        sp = root / "tune" / f"{fam}_selected.json"
        if not sp.exists():
            continue
        s = jload(sp)
        sel[fam] = s
        for p in sorted((root / "tune").glob(f"{fam}_cfg*.json")):
            r = jload(p)
            rows.append(dict(model=fam, lr=r["cfg"]["lr"], batch=r["cfg"]["bs"], weight_decay=r["cfg"]["wd"],
                             inner_val_macro_f1=f"{r['val_macro_f1_mean']:.3f} ± {r['val_macro_f1_sd']:.3f}",
                             best_epochs="/".join(str(e) for e in r["best_epochs"]),
                             selected="yes" if r["cfg_index"] == s["cfg_index"] else ""))
    return rows, sel


def analyse_gradcam(root):
    p = root / "gradcam" / "gradcam_cnn_ce_s0.npz"
    if not p.exists():
        return None
    g = np.load(p)
    y, pred = g["y"], g["pred"]
    names = [str(n) for n in g["region_names"]]
    valid = g["valid_true"] if "valid_true" in g.files else np.ones(len(y), bool)
    frac = g["frac_true"]
    rows = []
    for c, cname in enumerate(CLS):
        for subset in ("all", "correct", "misclassified"):
            m = (y == c) & valid
            if subset == "correct":
                m &= pred == c
            elif subset == "misclassified":
                m &= pred != c
            row = dict(cls=cname, subset=subset, n=int(m.sum()), n_total=int(((y == c) & (
                (pred == c) if subset == "correct" else (pred != c) if subset == "misclassified" else True)).sum()))
            for j, rn in enumerate(names):
                if m.any():
                    row[rn] = f"{frac[m, j].mean():.3f} ± {frac[m, j].std():.3f}"
                    row[rn + "_mean"] = float(frac[m, j].mean())
                else:
                    row[rn], row[rn + "_mean"] = "n/a", None
            rows.append(row)
    return dict(rows=rows, regions=names, region_bounds=g["region_bounds"].tolist(), file=str(p))


# ------------------------------------------------------------------ figures
def figures(root, outdir, ds, cells, effects, gc, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    blue, lblue, red, lred = "#1F4E79", "#9DC3E6", "#C00000", "#F4B183"

    # --- Figure 2: matched protocol comparison (2 x 2) ---
    if cells:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
        ax = axs[0]
        x = np.arange(2)
        w = 0.36
        for i, (fam, lab, col) in enumerate((("cnn", "1D-CNN (morphology)", blue), ("cnnrr", "1D-CNN + RR", lblue))):
            ms, lo, hi = [], [], []
            for proto in ("patient", "beat"):
                c = cells.get(f"{fam}_{proto}")
                ms.append(c["macro_f1_mean"] if c else np.nan)
                lo.append(c["macro_f1_mean"] - c["macro_f1_ci"][0] if c else 0)
                hi.append(c["macro_f1_ci"][1] - c["macro_f1_mean"] if c else 0)
            ax.bar(x + (i - 0.5) * w, ms, w, yerr=[lo, hi], capsize=3, color=col, label=lab, edgecolor="white")
        ax.set_xticks(x, ["Patient-wise CV\n(fair)", "Beat-wise CV\n(leaky)"])
        ax.set_ylabel("Macro-F1 (pooled out-of-fold)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_title("A  Macro-F1, 4 classes", loc="left", fontsize=9)
        ax = axs[1]
        x = np.arange(K)
        for i, (proto, col, lab) in enumerate((("patient", blue, "Patient-wise CV"), ("beat", red, "Beat-wise CV"))):
            c = cells.get(f"cnn_{proto}")
            if not c:
                continue
            m = np.array(c["f1_mean"])
            lo = m - np.array([v[0] for v in c["f1_ci"]])
            hi = np.array([v[1] for v in c["f1_ci"]]) - m
            ax.bar(x + (i - 0.5) * w, m, w, yerr=[lo, hi], capsize=3, color=col, label=lab, edgecolor="white")
        ax.set_xticks(x, CLS)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 (primary 1D-CNN)")
        ax.set_xlabel("AAMI class")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.set_title("B  Per-class F1, primary model", loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / "Figure2_protocol_comparison.png", dpi=300)
        plt.close(fig)

    # --- Figure 3: confusion matrices, primary CNN, patient- vs beat-wise CV ---
    if cells.get("cnn_patient") and cells.get("cnn_beat"):
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
        for ax, proto, title in ((axs[0], "patient", "Patient-wise 5-fold CV"), (axs[1], "beat", "Beat-wise 5-fold CV")):
            cm = np.array(cells[f"cnn_{proto}"]["cm_pooled_over_seeds"], float)
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(K), CLS)
            ax.set_yticks(range(K), CLS)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title, fontsize=9)
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f"{cmn[i, j] * 100:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else "#222")
        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, label="Row-normalized proportion")
        fig.savefig(outdir / "Figure3_confusion_matrices.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Figure 4: Grad-CAM class-mean profiles ---
    gp = root / "gradcam" / "gradcam_cnn_ce_s0.npz"
    if gp.exists():
        g = np.load(gp)
        prof = g["prof_true_all"]
        cnt = g["count_true_all"]
        bm, bs = g["beat_mean"], g["beat_sd"]
        t = (np.arange(C.BEAT_LEN) - C.R_IDX) / C.FS * 1000
        fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3), sharey=True)
        for c, ax in enumerate(axs):
            ax.fill_between(t, bm[c] - bs[c], bm[c] + bs[c], color="#D9D9D9", lw=0)
            ax.plot(t, bm[c], color="black", lw=0.9)
            ax2 = ax.twinx()
            ax2.fill_between(t, 0, prof[c], color=red, alpha=0.35, lw=0)
            ax2.plot(t, prof[c], color=red, lw=0.8)
            ax2.set_ylim(0, 1.05)
            ax2.set_yticks([] if c < 3 else [0, 0.5, 1])
            ax.set_zorder(ax2.get_zorder() + 1)      # draw the beat above the saliency fill
            ax.patch.set_visible(False)
            if c == 3:
                ax2.set_ylabel("Mean Grad-CAM (normalized)", color=red)
            for a, b in list(C.REGIONS.values())[1:]:
                ax.axvline((a - C.R_IDX) / C.FS * 1000, color="#7F7F7F", lw=0.4, ls="--")
            ax.set_title(f"{CLS[c]} (n={int(cnt[c])})", fontsize=8)
            ax.set_xlabel("Time from R peak (ms)")
            if c == 0:
                ax.set_ylabel("Mean beat (z-score)")
        fig.tight_layout()
        fig.savefig(outdir / "Figure4_gradcam_class_means.png", dpi=300)
        plt.close(fig)

    # --- Supplementary: per-class Se / PPV / F1 on DS2 for all configurations ---
    if ds:
        names = [n for n in DS_ORDER if n in ds]
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True)
        cols = plt.cm.tab10(np.linspace(0, 1, len(names)))
        for ax, key, title in zip(axs, ("se", "ppv", "f1"), ("Sensitivity", "Positive predictive value", "F1")):
            x = np.arange(K)
            w = 0.8 / len(names)
            for i, n in enumerate(names):
                ax.bar(x - 0.4 + w / 2 + i * w, ds[n][f"{key}_mean"], w, yerr=ds[n][f"{key}_sd"], capsize=1,
                       color=cols[i], label=n, error_kw=dict(lw=0.5))
            ax.set_xticks(x, CLS)
            ax.set_title(title, fontsize=9)
            ax.set_ylim(0, 1.05)
        axs[0].legend(frameon=False, fontsize=5.5, loc="upper right")
        fig.tight_layout()
        fig.savefig(outdir / "FigureS2_DS2_perclass_all_configs.png", dpi=300)
        plt.close(fig)

        gam = [g for g in (0.5, 1, 2, 3, 5) if f"cnn_focal_g{g:g}" in ds]
        if gam:
            fig, ax = plt.subplots(figsize=(3.6, 2.6))
            for key, cls, lab, col in (("macro_f1", None, "Macro-F1", blue), ("se", 1, "S sensitivity", red),
                                       ("se", 3, "F sensitivity", "#548235")):
                if cls is None:
                    m = [ds[f"cnn_focal_g{g:g}"]["macro_f1_mean"] for g in gam]
                    s = [ds[f"cnn_focal_g{g:g}"]["macro_f1_sd"] for g in gam]
                else:
                    m = [ds[f"cnn_focal_g{g:g}"]["se_mean"][cls] for g in gam]
                    s = [ds[f"cnn_focal_g{g:g}"]["se_sd"][cls] for g in gam]
                ax.errorbar(gam, m, yerr=s, marker="o", ms=3, capsize=2, color=col, label=lab, lw=1)
            ax.set_xlabel("Focal-loss focusing parameter γ")
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig(outdir / "FigureS3_focal_gamma.png", dpi=300)
            plt.close(fig)

    # --- Supplementary: training curves ---
    fams = [f for f in ("cnn", "cnnrr", "resnet") if f in sel]
    if fams:
        fig, axs = plt.subplots(2, len(fams), figsize=(2.6 * len(fams), 4.0), squeeze=False)
        for j, fam in enumerate(fams):
            s = sel[fam]
            r = jload(root / "tune" / f"{fam}_cfg{s['cfg_index']:02d}.json")
            for curve in r["curves"]:
                e = [c["epoch"] for c in curve]
                axs[0, j].plot(e, [c["train_loss"] for c in curve], lw=0.8)
                axs[1, j].plot(e, [c.get("val_macro_f1", np.nan) for c in curve], lw=0.8)
            axs[1, j].axvline(s["epochs"], color="k", ls="--", lw=0.6)
            axs[0, j].set_title({"cnn": "1D-CNN", "cnnrr": "1D-CNN + RR", "resnet": "1D-ResNet"}[fam], fontsize=9)
            axs[1, j].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Training loss (inner folds)")
        axs[1, 0].set_ylabel("Inner-validation macro-F1")
        fig.tight_layout()
        fig.savefig(outdir / "FigureS1_training_curves.png", dpi=300)
        plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.OUT))
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    summ = dict(B=a.B, root=str(root))
    if (root / "data_summary.json").exists():
        summ["data"] = jload(root / "data_summary.json")
        write_csv(out / "tables" / "TableS1_records.csv", summ["data"]["per_record"])
    hpo_rows, sel = analyse_tuning(root)
    summ["tuning_selected"] = sel
    write_csv(out / "tables" / "TableS2_hyperparameter_search.csv", hpo_rows)

    ds, dsinfo = analyse_ds(root, a.B) if (root / "ds" / "labels.npz").exists() else ({}, {})
    summ["ds"], summ["ds_info"] = ds, dsinfo
    rows = []
    for n in DS_ORDER:
        if n not in ds:
            continue
        r = ds[n]
        rows.append(dict(config=n, label=r["label"], seeds=r["n_seeds"],
                         accuracy=fmt(r["acc_mean"], r["acc_sd"]),
                         accuracy_95CI=f"{r['acc_ci'][0]:.3f}-{r['acc_ci'][1]:.3f}",
                         macro_F1=fmt(r["macro_f1_mean"], r["macro_f1_sd"]),
                         macro_F1_95CI=f"{r['macro_f1_ci'][0]:.3f}-{r['macro_f1_ci'][1]:.3f}",
                         delta_vs_primary=("" if n == "cnn_ce" else f"{r['delta_macro_f1_vs_primary']:+.3f}"),
                         delta_95CI=("" if n == "cnn_ce" else "{:+.3f} to {:+.3f}".format(*r["delta_macro_f1_vs_primary_ci"])),
                         delta_p=("" if n == "cnn_ce" else f"{r['delta_macro_f1_vs_primary_p']:.3f}")))
    write_csv(out / "tables" / "Table2_DS2_overall.csv", rows)
    rows = []
    for n in DS_ORDER:
        if n not in ds:
            continue
        r = ds[n]
        for c, cn in enumerate(CLS):
            rows.append(dict(config=n, cls=cn, support=int(r["support"][c]),
                             Se=fmt(r["se_mean"][c], r["se_sd"][c]), PPV=fmt(r["ppv_mean"][c], r["ppv_sd"][c]),
                             F1=fmt(r["f1_mean"][c], r["f1_sd"][c])))
    write_csv(out / "tables" / "Table3_DS2_perclass.csv", rows)

    cells, effects, cvinfo = analyse_cv(root, a.B) if (root / "cv" / "labels.npz").exists() else ({}, {}, {})
    summ["cv"], summ["cv_effects"], summ["cv_info"] = cells, effects, cvinfo
    rows = []
    for key, c in cells.items():
        rows.append(dict(model=c["family"], protocol=c["protocol"], seeds=c["n_seeds"],
                         pooled_macro_F1=fmt(c["macro_f1_mean"], c["macro_f1_sd"]),
                         pooled_macro_F1_95CI=f"{c['macro_f1_ci'][0]:.3f}-{c['macro_f1_ci'][1]:.3f}",
                         foldwise_macro_F1=fmt(c["foldwise_macro_f1_mean"], c["foldwise_macro_f1_sd"]),
                         accuracy=fmt(c["acc_mean"], c["acc_sd"]),
                         **{f"{cn}_Se": fmt(c["se_mean"][i], c["se_sd"][i]) for i, cn in enumerate(CLS)},
                         **{f"{cn}_PPV": fmt(c["ppv_mean"][i], c["ppv_sd"][i]) for i, cn in enumerate(CLS)},
                         **{f"{cn}_F1": fmt(c["f1_mean"][i], c["f1_sd"][i]) for i, cn in enumerate(CLS)}))
    write_csv(out / "tables" / "Table4_matched_CV.csv", rows)
    write_csv(out / "tables" / "Table4b_effects.csv",
              [dict(effect=k, point=f"{v['point']:+.3f}" if "point" in v else "",
                    CI95=("{:+.3f} to {:+.3f}".format(*v["ci"]) if "ci" in v else ""),
                    p=f"{v['p']:.4f}", t=f"{v.get('t', float('nan')):.3f}") for k, v in effects.items()])

    gc = analyse_gradcam(root)
    summ["gradcam"] = gc
    if gc:
        write_csv(out / "tables" / "Table5_gradcam_regions.csv",
                  [{k: v for k, v in r.items() if not k.endswith("_mean")} for r in gc["rows"]])
    if (root / "efficiency.json").exists():
        e = jload(root / "efficiency.json")
        summ["efficiency"] = e
        write_csv(out / "tables" / "Table6_efficiency.csv", e["models"])
    figures(root, out / "figures", ds, cells, effects, gc, sel)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"Analysis written to {out}")
    if ds.get("cnn_ce"):
        r = ds["cnn_ce"]
        print(f"Primary DS2: acc {fmt(r['acc_mean'], r['acc_sd'])}, macro-F1 {fmt(r['macro_f1_mean'], r['macro_f1_sd'])} "
              f"(95% CI {r['macro_f1_ci'][0]:.3f}-{r['macro_f1_ci'][1]:.3f})")
    for k in ("protocol_cnn_macro_f1", "protocol_cnnrr_macro_f1", "model_patient_macro_f1", "interaction_macro_f1"):
        if k in effects:
            v = effects[k]
            print(f"{k}: {v['point']:+.3f} (95% CI {v['ci'][0]:+.3f} to {v['ci'][1]:+.3f}), p={v['p']:.4f}")


if __name__ == "__main__":
    main()
