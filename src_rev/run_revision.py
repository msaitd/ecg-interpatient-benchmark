"""
Experiment orchestrator.

  python src_rev/run_revision.py            full run (a CUDA GPU is strongly recommended)
  python src_rev/run_revision.py --smoke    tiny end-to-end check (minutes, CPU is fine)
  python src_rev/run_revision.py --stages tune,ds

Stages (each run is saved to results_rev/; finished runs are skipped, so the script is
resumable after an interruption):
  data     data summary (class counts, per-record counts, record splits, class weights)
  tune     hyper-parameter tuning by inner patient-wise 3-fold CV on DS1 (cnn, cnnrr, resnet)
  ds       DS1 -> DS2 benchmark, 5 seeds: CNN (CE, class weights, oversampling, focal gamma
           grid), CNN+RR, 1D-ResNet, random forest
  cv       matched protocol comparison on the pooled 44 records: patient-wise vs beat-wise
           5-fold CV, CNN and CNN+RR, 5 seeds
  gradcam  Grad-CAM of the primary CNN on all DS2 beats (class-mean maps, region fractions)
  eff      parameters, model size, CPU latency, and hardware information
  summary  short printout of the main numbers
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rconfig as C  # noqa: E402

LOGF = None


def log(msg=""):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOGF:
        with open(LOGF, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_conv)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ hardware
def hardware_info():
    import torch
    import sklearn
    info = dict(os=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__)
    cpu = platform.processor()
    try:
        if sys.platform.startswith("win"):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        elif os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    info["cpu"] = cpu
    info["logical_cpus"] = os.cpu_count()
    ram = None
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            ram = ms.ullTotalPhys / 1024 ** 3
        elif os.path.exists("/proc/meminfo"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    ram = int(line.split()[1]) / 1024 ** 2
                    break
    except Exception:
        pass
    info["ram_GB"] = round(ram, 1) if ram else None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["gpu_mem_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    else:
        info["gpu"] = None
    return info


# ------------------------------------------------------------------ stages
def stage_data(D):
    import rdata
    p = C.OUT / "data_summary.json"
    rows = []
    for split, recs in (("train", C.DS1), ("test", C.DS2)):
        d = D[split]
        for r in recs:
            m = d["rec"] == r
            rows.append(dict(split="DS1" if split == "train" else "DS2", record=r,
                             **{c: int((d["y"][m] == i).sum()) for i, c in enumerate(C.CLASSES)}))
    summ = dict(classes=C.CLASSES, DS1=C.DS1, DS2=C.DS2,
                counts_DS1=rdata.class_counts(D["train"]["y"]).tolist(),
                counts_DS2=rdata.class_counts(D["test"]["y"]).tolist(),
                class_weights_DS1=rdata.inverse_frequency_weights(D["train"]["y"]).tolist(),
                per_record=rows)
    jdump(summ, p)
    log(f"data: DS1 {summ['counts_DS1']}  DS2 {summ['counts_DS2']}  "
        f"class weights {np.round(summ['class_weights_DS1'], 3).tolist()}")


def stage_tune(D, device, family):
    import rdata, rmodels, rtrain
    from sklearn.model_selection import StratifiedGroupKFold
    sel_path = C.OUT / "tune" / f"{family}_selected.json"
    if sel_path.exists():
        sel = jload(sel_path)
        log(f"tune[{family}] already done: {sel['cfg']} epochs={sel['epochs']}")
        return sel
    tr = D["train"]
    X, y, rec, rr = tr["X"], tr["y"], tr["rec"], tr["rr"]
    grid = C.GRID_RESNET if family == "resnet" else C.GRID_CNN
    folds = list(StratifiedGroupKFold(n_splits=C.INNER_FOLDS, shuffle=True, random_state=0)
                 .split(np.zeros(len(y)), y, groups=rec))
    rows = []
    for gi, g in enumerate(grid):
        p = C.OUT / "tune" / f"{family}_cfg{gi:02d}.json"
        if p.exists():
            rows.append(jload(p))
            continue
        scores, eps, curves, t0 = [], [], [], time.time()
        for fi, (a, b) in enumerate(folds):
            if family == "cnnrr":
                rr_a, rr_b = rdata.standardize(rr[a], rr[b])
            else:
                rr_a = rr_b = None
            m = rmodels.build(family)
            _, curve, bep, best = rtrain.train_model(
                m, X[a], y[a], rr_a, device, seed=0, lr=g["lr"], bs=g["bs"], wd=g["wd"],
                epochs=C.MAX_EPOCHS_TUNE, kind="ce", Xval=X[b], yval=y[b], rrval=rr_b,
                patience=C.PATIENCE)
            scores.append(best)
            eps.append(bep)
            curves.append(curve)
        row = dict(family=family, cfg_index=gi, cfg=g,
                   val_macro_f1_mean=float(np.mean(scores)),
                   val_macro_f1_sd=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                   fold_scores=scores, best_epochs=eps, curves=curves)
        jdump(row, p)
        rows.append(row)
        log(f"tune[{family}] cfg {gi + 1}/{len(grid)} {g} -> inner-val macro-F1 "
            f"{row['val_macro_f1_mean']:.3f} (best epochs {eps}) [{time.time() - t0:.0f}s]")
    best = max(rows, key=lambda r: r["val_macro_f1_mean"])
    epochs = max(C.MIN_EPOCHS, int(round(float(np.median(best["best_epochs"])))))
    sel = dict(family=family, cfg=best["cfg"], cfg_index=best["cfg_index"], epochs=epochs,
               val_macro_f1_mean=best["val_macro_f1_mean"], val_macro_f1_sd=best["val_macro_f1_sd"],
               inner_fold_records=[sorted({int(v) for v in rec[b]}) for _, b in folds])
    jdump(sel, sel_path)
    log(f"tune[{family}] SELECTED {sel['cfg']} epochs={epochs} "
        f"(inner-val macro-F1 {sel['val_macro_f1_mean']:.3f})")
    return sel


def ds_configs():
    cfgs = [("cnn_ce", "cnn", "ce", None), ("cnn_cw", "cnn", "cw", None), ("cnn_os", "cnn", "os", None)]
    cfgs += [(f"cnn_focal_g{g:g}", "cnn", "focal", g) for g in C.FOCAL_GAMMAS]
    cfgs += [("cnnrr_ce", "cnnrr", "ce", None), ("resnet_ce", "resnet", "ce", None)]
    return cfgs


def stage_ds(D, device, sel):
    import torch
    import rdata, rmodels, rtrain, rmetrics
    tr, te = D["train"], D["test"]
    d = C.OUT / "ds"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "labels.npz", y=te["y"], rec=te["rec"])
    (C.OUT / "models").mkdir(parents=True, exist_ok=True)
    cfgs = ds_configs()
    total = len(cfgs) * len(C.SEEDS) + len(C.SEEDS)
    k = 0
    for name, fam, kind, gamma in cfgs:
        s = sel[fam]
        g = s["cfg"]
        for seed in C.SEEDS:
            k += 1
            pj = d / f"{name}_s{seed}.json"
            if pj.exists():
                continue
            if fam == "cnnrr":
                rr_tr, rr_te = rdata.standardize(tr["rr"], te["rr"])
            else:
                rr_tr = rr_te = None
            t0 = time.time()
            m = rmodels.build(fam)
            m, curve, _, _ = rtrain.train_model(m, tr["X"], tr["y"], rr_tr, device, seed, g["lr"],
                                                g["bs"], g["wd"], s["epochs"], kind=kind,
                                                gamma=gamma if gamma is not None else 2.0)
            P = rtrain.predict_proba(m, te["X"], rr_te, device)
            np.savez_compressed(d / f"{name}_s{seed}.npz", probs=P.astype(np.float16))
            met = rmetrics.summarize(te["y"], P.argmax(1))
            meta = dict(name=name, family=fam, loss=kind, gamma=gamma, seed=seed, cfg=g,
                        epochs=s["epochs"], params=rmodels.n_params(m), curve=curve, metrics=met,
                        train_seconds=round(time.time() - t0, 1))
            if kind == "cw":
                meta["class_weights"] = rdata.inverse_frequency_weights(tr["y"]).tolist()
            if kind == "os":
                meta["oversampling"] = "class-balanced, p(beat) proportional to 1/n_class, with replacement"
            jdump(meta, pj)
            if seed == C.SEEDS[0] and kind == "ce":
                torch.save(m.state_dict(), C.OUT / "models" / f"{name}_s{seed}.pt")
            log(f"ds [{k}/{total}] {name} seed {seed}: acc {met['acc']:.3f}  macro-F1 "
                f"{met['macro_f1']:.3f}  S-Se {met['se'][1]:.3f}  ({time.time() - t0:.0f}s)")
    from sklearn.ensemble import RandomForestClassifier
    for seed in C.SEEDS:
        k += 1
        pj = d / f"rf_s{seed}.json"
        if pj.exists():
            continue
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=C.RF_TREES, class_weight="balanced",
                                    n_jobs=-1, random_state=seed).fit(tr["rf"], tr["y"])
        P = np.zeros((len(te["y"]), C.K), np.float32)
        P[:, rf.classes_] = rf.predict_proba(te["rf"])
        np.savez_compressed(d / f"rf_s{seed}.npz", probs=P.astype(np.float16))
        met = rmetrics.summarize(te["y"], P.argmax(1))
        jdump(dict(name="rf", family="rf", seed=seed, trees=C.RF_TREES, metrics=met,
                   train_seconds=round(time.time() - t0, 1)), pj)
        log(f"ds [{k}/{total}] rf seed {seed}: acc {met['acc']:.3f}  macro-F1 {met['macro_f1']:.3f}")


def stage_cv(D, device, sel):
    import rdata, rmodels, rtrain, rmetrics
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    tr, te = D["train"], D["test"]
    X = np.concatenate([tr["X"], te["X"]])
    y = np.concatenate([tr["y"], te["y"]])
    rec = np.concatenate([tr["rec"], te["rec"]])
    rr = np.concatenate([tr["rr"], te["rr"]])
    n = len(y)
    d = C.OUT / "cv"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "labels.npz", y=y, rec=rec)
    total = 2 * 2 * len(C.SEEDS)
    k = 0
    for fam in ("cnn", "cnnrr"):
        s = sel[fam]
        g = s["cfg"]
        for proto in ("patient", "beat"):
            for seed in C.SEEDS:
                k += 1
                key = f"{fam}_ce_{proto}_s{seed}"
                pj = d / f"{key}.json"
                if pj.exists():
                    continue
                if proto == "patient":
                    folds = list(StratifiedGroupKFold(n_splits=C.N_FOLDS_CV, shuffle=True,
                                                      random_state=seed).split(np.zeros(n), y, groups=rec))
                else:
                    folds = list(StratifiedKFold(n_splits=C.N_FOLDS_CV, shuffle=True,
                                                 random_state=seed).split(np.zeros(n), y))
                oof = np.zeros((n, C.K), np.float32)
                fid = np.full(n, -1, np.int64)
                fmeta, t0 = [], time.time()
                for fi, (a, b) in enumerate(folds):
                    if fam == "cnnrr":
                        rr_a, rr_b = rdata.standardize(rr[a], rr[b])
                    else:
                        rr_a = rr_b = None
                    m = rmodels.build(fam)
                    m, curve, _, _ = rtrain.train_model(m, X[a], y[a], rr_a, device, seed * 100 + fi,
                                                        g["lr"], g["bs"], g["wd"], s["epochs"], kind="ce")
                    P = rtrain.predict_proba(m, X[b], rr_b, device)
                    oof[b] = P
                    fid[b] = fi
                    fmeta.append(dict(fold=fi, n_train=int(len(a)), n_test=int(len(b)),
                                      test_records=sorted({int(v) for v in rec[b]}) if proto == "patient" else None,
                                      metrics=rmetrics.summarize(y[b], P.argmax(1)),
                                      final_train_loss=curve[-1]["train_loss"]))
                np.savez_compressed(d / f"{key}.npz", probs=oof.astype(np.float16), fold=fid)
                pm = rmetrics.summarize(y, oof.argmax(1))
                jdump(dict(name=f"{fam}_ce", family=fam, protocol=proto, seed=seed, cfg=g,
                           epochs=s["epochs"], pooled_metrics=pm, folds=fmeta,
                           seconds=round(time.time() - t0, 1)), pj)
                log(f"cv [{k}/{total}] {fam} {proto}-wise seed {seed}: pooled macro-F1 "
                    f"{pm['macro_f1']:.3f}  S-Se {pm['se'][1]:.3f}  ({time.time() - t0:.0f}s)")


def stage_gradcam(D, device):
    import torch
    import torch.nn.functional as F
    import rmodels
    p = C.OUT / "gradcam" / "gradcam_cnn_ce_s0.npz"
    if p.exists():
        log("gradcam already done")
        return
    te = D["test"]
    X, y = te["X"], te["y"]
    m = rmodels.build("cnn")
    m.load_state_dict(torch.load(C.OUT / "models" / f"cnn_ce_s{C.SEEDS[0]}.pt", map_location=device))
    m.to(device).eval()
    n, L = len(y), C.BEAT_LEN
    pred = np.zeros(n, np.int64)
    cams = {"true": np.zeros((n, L), np.float32), "pred": np.zeros((n, L), np.float32)}
    with torch.enable_grad():
        for s in range(0, n, 1024):
            xb = torch.as_tensor(X[s:s + 1024], device=device)
            yb = torch.as_tensor(y[s:s + 1024], device=device)
            for tgt_name in ("true", "pred"):
                logits, A = m(xb, None, return_feats=True)
                if tgt_name == "pred":
                    tgt = logits.argmax(1)
                    pred[s:s + 1024] = tgt.cpu().numpy()
                else:
                    tgt = yb
                score = logits.gather(1, tgt[:, None]).sum()
                G = torch.autograd.grad(score, A)[0]
                cam = F.relu((G.mean(-1, keepdim=True) * A).sum(1))
                cam = F.interpolate(cam[:, None, :], size=L, mode="linear", align_corners=False)[:, 0]
                cams[tgt_name][s:s + 1024] = cam.detach().cpu().numpy()
    out = dict(y=y, pred=pred, region_names=np.array(list(C.REGIONS)),
               region_bounds=np.array(list(C.REGIONS.values())))
    for tname, cam in cams.items():
        valid = cam.max(1) > 0                   # all-zero maps (after ReLU) carry no information
        out[f"valid_{tname}"] = valid
        mass = cam / (cam.sum(1, keepdims=True) + 1e-12)
        out[f"frac_{tname}"] = np.stack([mass[:, a:b].sum(1) for a, b in C.REGIONS.values()], 1).astype(np.float32)
        peak = cam / (cam.max(1, keepdims=True) + 1e-12)
        for subset in ("all", "correct", "wrong"):
            prof = np.zeros((C.K, L), np.float32)
            cnt = np.zeros(C.K, np.int64)
            for c in range(C.K):
                msk = (y == c) & valid
                if subset == "correct":
                    msk &= pred == c
                elif subset == "wrong":
                    msk &= pred != c
                cnt[c] = msk.sum()
                if cnt[c]:
                    prof[c] = peak[msk].mean(0)
            out[f"prof_{tname}_{subset}"] = prof
            out[f"count_{tname}_{subset}"] = cnt
    Xb = X[:, 0, :]
    out["beat_mean"] = np.stack([Xb[y == c].mean(0) if (y == c).any() else np.zeros(L) for c in range(C.K)])
    out["beat_sd"] = np.stack([Xb[y == c].std(0) if (y == c).any() else np.zeros(L) for c in range(C.K)])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    log(f"gradcam saved ({n} DS2 beats)")


def stage_eff(smoke):
    import torch
    import rmodels
    rows = []
    for fam, name in (("cnn", "cnn_ce"), ("cnnrr", "cnnrr_ce"), ("resnet", "resnet_ce")):
        m = rmodels.build(fam)
        wp = C.OUT / "models" / f"{name}_s{C.SEEDS[0]}.pt"
        if wp.exists():
            m.load_state_dict(torch.load(wp, map_location="cpu"))
        m = m.cpu().eval()
        nthreads = torch.get_num_threads()
        torch.set_num_threads(1)
        x1, xb = torch.randn(1, 1, C.BEAT_LEN), torch.randn(256, 1, C.BEAT_LEN)
        r1 = torch.randn(1, 5) if fam == "cnnrr" else None
        rb = torch.randn(256, 5) if fam == "cnnrr" else None
        reps1, repsb = (20, 3) if smoke else (1000, 30)
        with torch.no_grad():
            for _ in range(20):
                m(x1, r1)
            t0 = time.perf_counter()
            for _ in range(reps1):
                m(x1, r1)
            single = (time.perf_counter() - t0) / reps1 * 1000
            for _ in range(2):
                m(xb, rb)
            t0 = time.perf_counter()
            for _ in range(repsb):
                m(xb, rb)
            per_beat = (time.perf_counter() - t0) / (repsb * 256) * 1000
        torch.set_num_threads(nthreads)
        npar = rmodels.n_params(m)
        rows.append(dict(model=name, params=npar, size_MB=round(npar * 4 / 1024 ** 2, 4),
                         latency_ms_single_beat=round(single, 4),
                         latency_ms_per_beat_batch256=round(per_beat, 5),
                         throughput_beats_per_s=round(1000 / per_beat, 1)))
        log(f"eff {name}: {npar} params, {single:.3f} ms/beat (single thread)")
    jdump(dict(hardware=hardware_info(), threads_used_for_latency=1, models=rows), C.OUT / "efficiency.json")


def stage_summary():
    import rmetrics
    d = C.OUT / "ds"
    log("=" * 60)
    log("SUMMARY (DS1->DS2, mean over seeds)")
    names = sorted({p.name.rsplit("_s", 1)[0] for p in d.glob("*_s*.json")})
    for nm in names:
        ms = [jload(p)["metrics"] for p in sorted(d.glob(f"{nm}_s*.json"))]
        f1 = [m["macro_f1"] for m in ms]
        sse = [m["se"][1] for m in ms]
        log(f"  {nm:<18} macro-F1 {np.mean(f1):.3f} +/- {np.std(f1, ddof=1) if len(f1) > 1 else 0:.3f}"
            f"   S-Se {np.mean(sse):.3f}   (n={len(ms)})")
    d = C.OUT / "cv"
    log("MATCHED 5-FOLD CV (pooled out-of-fold macro-F1, mean over seeds)")
    for fam in ("cnn", "cnnrr"):
        for proto in ("patient", "beat"):
            ms = [jload(p)["pooled_metrics"]["macro_f1"] for p in sorted(d.glob(f"{fam}_ce_{proto}_s*.json"))]
            if ms:
                log(f"  {fam:<6} {proto:<8} macro-F1 {np.mean(ms):.3f} +/- "
                    f"{np.std(ms, ddof=1) if len(ms) > 1 else 0:.3f} (n={len(ms)})")
    log("=" * 60)


# ------------------------------------------------------------------ main
def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stages", default="data,tune,ds,cv,gradcam,eff,summary")
    args = ap.parse_args()
    if args.smoke:
        C.apply_smoke()
    C.OUT.mkdir(parents=True, exist_ok=True)
    LOGF = C.OUT / "revision_log.txt"
    import rdata
    import rtrain
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    device = rtrain.get_device()
    hw = hardware_info()
    log("=" * 60)
    log(f"Revision run started | smoke={args.smoke} | device={device} | stages={stages}")
    log(f"Hardware: CPU={hw['cpu']} | RAM={hw['ram_GB']} GB | GPU={hw['gpu']} | torch={hw['torch']}")
    if device.type == "cpu" and not args.smoke:
        log("WARNING: no CUDA GPU detected - the full run will be very slow on CPU.")
    t_all = time.time()
    D = rdata.load_all(subsample=3000 if args.smoke else None)
    log(f"Data loaded: DS1 {len(D['train']['y'])} beats, DS2 {len(D['test']['y'])} beats (4 classes, Q excluded)")
    sel = {}
    try:
        if "data" in stages:
            stage_data(D)
        if any(s in stages for s in ("tune", "ds", "cv")):
            for fam in ("cnn", "cnnrr", "resnet"):
                if fam == "resnet" and "tune" not in stages and "ds" not in stages:
                    continue
                sel[fam] = stage_tune(D, device, fam)
        if "ds" in stages:
            stage_ds(D, device, sel)
        if "cv" in stages:
            stage_cv(D, device, sel)
        if "gradcam" in stages:
            stage_gradcam(D, device)
        if "eff" in stages:
            stage_eff(args.smoke)
        if "summary" in stages:
            stage_summary()
    except Exception:
        log("ERROR:\n" + traceback.format_exc())
        log("The run stopped. Fix the problem and start it again: finished runs are kept and skipped.")
        sys.exit(1)
    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min. Results are in {C.OUT}")


if __name__ == "__main__":
    main()
