"""
Step 02 — Preprocess MIT-BIH into AAMI-labelled heartbeats (inter-patient split).

For every record:
  * select the MLII lead (fall back to the first lead if absent),
  * band-pass filter the signal,
  * for each annotated beat: take a fixed window around the R-peak,
    map the annotation symbol to one of the 5 AAMI classes,
    compute pre/post RR intervals (seconds),
  * collect beats into DS1 (train) and DS2 (test).

Outputs:  data/processed/train.npz  and  data/processed/test.npz
          each with arrays  X (beats), y (labels), rec (record id), rr (pre,post)

Run:  python src/02_preprocess.py
"""
import numpy as np
import wfdb

import config as C
from utils import bandpass


def process_record(rec_id: int):
    path = str(C.DATA_RAW / str(rec_id))
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    # ---- choose lead ------------------------------------------------------
    names = list(record.sig_name)
    lead = names.index(C.PREFERRED_LEAD) if C.PREFERRED_LEAD in names else 0
    sig = record.p_signal[:, lead].astype(np.float64)
    sig = bandpass(sig)

    # ---- keep only beat annotations mapped to an AAMI class ---------------
    beats = [(int(s), C.AAMI[sym])
             for s, sym in zip(ann.sample, ann.symbol) if sym in C.AAMI]

    X, y, rr = [], [], []
    for k, (r, cls) in enumerate(beats):
        a, b = r - C.WIN_LEFT, r + C.WIN_RIGHT
        if a < 0 or b > len(sig):
            continue
        pre = (r - beats[k - 1][0]) / C.FS if k > 0 else 0.0
        post = (beats[k + 1][0] - r) / C.FS if k < len(beats) - 1 else 0.0
        X.append(sig[a:b].astype(np.float32))
        y.append(C.CLASS_TO_IDX[cls])
        rr.append([pre, post])
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.int8),
            np.full(len(y), rec_id, dtype=np.int16),
            np.asarray(rr, dtype=np.float32))


def build(records, name):
    Xs, ys, recs, rrs = [], [], [], []
    for r in records:
        if not (C.DATA_RAW / f"{r}.dat").exists():
            print(f"  [skip] record {r} not found in data/raw (run 01 first)")
            continue
        X, y, rec, rr = process_record(r)
        Xs.append(X); ys.append(y); recs.append(rec); rrs.append(rr)
        print(f"  record {r}: {len(y):>5} beats")
    X = np.concatenate(Xs); y = np.concatenate(ys)
    rec = np.concatenate(recs); rr = np.concatenate(rrs)
    out = C.DATA_PROC / f"{name}.npz"
    np.savez_compressed(out, X=X, y=y, rec=rec, rr=rr)
    dist = {C.CLASSES[i]: int((y == i).sum()) for i in range(len(C.CLASSES))}
    print(f"[{name}] {len(y)} beats  class distribution: {dist}")
    print(f"[{name}] saved -> {out}\n")


def main():
    print("Building DS1 (training) ...")
    build(C.DS1, "train")
    print("Building DS2 (testing) ...")
    build(C.DS2, "test")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
