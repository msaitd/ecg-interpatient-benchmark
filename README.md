# ECG inter-patient benchmark (MIT-BIH): lightweight 1D-CNN, matched evaluation protocols

Code accompanying the article

> **Fair Evaluation Matters: An Inter-Patient Benchmark of a Lightweight 1D Convolutional Neural
> Network for ECG Arrhythmia Classification on the MIT-BIH Database** (under review).

The code reproduces every number, table and figure of the revised article from the public
MIT-BIH Arrhythmia Database. No data are redistributed here; the scripts download them from
PhysioNet.

---

## 1. What the code does

| Item | Setting |
|---|---|
| Data | MIT-BIH Arrhythmia Database (PhysioNet `mitdb` 1.0.0), lead MLII, 360 Hz |
| Records | 44 non-paced records; paced records 102, 104, 107, 217 excluded (ANSI/AAMI EC57) |
| Split | de Chazal inter-patient split: DS1 (22 records, training) / DS2 (22 records, test), see `splits/records_DS1_DS2.csv` |
| Classes | 4 AAMI classes: N, S (SVEB), V (VEB), F (fusion). The 15 Q beats (8 in DS1, 7 in DS2) are excluded |
| Beat | 280 samples (0.78 s) centred on the annotated R peak, 0.5–40 Hz Butterworth band-pass, per-beat z-score |
| Primary model | 1D-CNN (3 conv blocks, 9,412 parameters), plain cross-entropy |
| Other models | class-weighted cross-entropy, focal loss (gamma 0.5/1/2/3/5, no alpha), class-balanced oversampling, CNN + 5 RR-interval features, 1D-ResNet (543,652 parameters), random forest (28 hand-crafted features) |
| Tuning | inner patient-wise 3-fold CV on DS1 only (grid over learning rate, batch size, weight decay; early stopping on inner-validation macro-F1); final models trained on all of DS1 for the median best epoch |
| Seeds | 5 (0–4); results as mean ± SD |
| Protocol comparison | pooled 44 records, 5-fold CV, **identical** folds procedure, models retrained in each fold: patient-wise (`StratifiedGroupKFold`, group = record) vs beat-wise (`StratifiedKFold`), CNN and CNN+RR |
| Statistics | record-level (patient-level) bootstrap, B = 2000, 95% percentile CIs; paired bootstrap tests for differences; paired t-test across seeds |
| Explainability | Grad-CAM on the last convolutional layer; fraction of attribution in pre-beat, P/PR, QRS and ST-T regions |
| Efficiency | parameters, model size, single-thread CPU latency per beat; CPU/RAM/GPU recorded automatically |

Macro-F1 is the unweighted mean of the per-class F1 over the classes that have at least one true
beat in the evaluated set (N, S, V, F). PPV is set to 0 for a class with no predicted beats and F1
to 0 when Se + PPV = 0.

---

## 2. Installation

Python 3.9 or newer (developed with Python 3.13). A CUDA GPU is strongly recommended for the full
revision run (about 1–2 h on a desktop GPU); the smoke test runs on a CPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch follow the selector at https://pytorch.org/get-started/locally/.

---

## 3. Reproducing the results

```bash
# 1) data (44 records, ~100 MB)
python src/01_download_data.py
#    if physionet.org cannot be reached, download the ZIP manually and run:
#    python src/01b_import_local.py "path/to/mit-bih-arrhythmia-database-1.0.0.zip"

# 2) beat extraction and AAMI labelling -> data/processed/train.npz (DS1), test.npz (DS2)
python src/02_preprocess.py

# 3) all experiments of the revised article (resumable) -> results_rev/
python src_rev/run_revision.py --smoke     # optional end-to-end check (minutes)
python src_rev/run_revision.py

# 4) statistics, tables and figures -> results_rev/analysis/
python src_rev/analyze_revision.py
```

On Windows, steps 3–4 are also run by double-clicking `run_revision.bat` (log:
`results_rev/revision_log.txt`). If a run is interrupted, start it again: finished runs are kept
and skipped.

Single stages can be run with `--stages`, e.g. `python src_rev/run_revision.py --stages tune,ds`.
Stages: `data, tune, ds, cv, gradcam, eff, summary`.

### Original (first-submission) pipeline

`src/03_*.py`–`src/07_*.py` and `run_pipeline.py` / `run_all.bat` reproduce the analysis of the
first submission (five AAMI classes, one seed, intra-patient 5-fold CV only). They are kept for
transparency; the revised article is based on `src_rev/`.

---

## 4. Outputs (`results_rev/`)

| Path | Content |
|---|---|
| `data_summary.json` | class counts per split, per-record counts, class weights |
| `tune/` | every tuning configuration (inner-fold macro-F1, best epochs) and the selected one |
| `ds/` | DS1→DS2 test probabilities and metrics for every configuration and seed, training curves |
| `cv/` | out-of-fold probabilities, fold assignments and metrics for the patient-wise and beat-wise CV |
| `gradcam/` | class-mean Grad-CAM profiles and region fractions |
| `efficiency.json` | parameters, size, latency, hardware |
| `analysis/tables/*.csv` | Tables 2–6 and Supplementary Tables S1–S2 of the article |
| `analysis/figures/*.png` | Figures 2–4 and Supplementary Figures S1–S3 (300 dpi) |
| `analysis/summary.json` | all statistics used in the text |

GPU training is not bit-wise deterministic across hardware and library versions; small
differences in the third decimal are expected.

---

## 5. Repository layout

```
src/              data download, preprocessing (01, 01b, 02) and the first-submission pipeline
src_rev/          revision experiments: config, data, models, training, metrics, orchestrator, analysis
splits/           DS1/DS2 record lists
run_revision.bat  Windows launcher for the revision experiments (.sh for macOS/Linux)
run_all.bat       Windows launcher for the first-submission pipeline (.sh for macOS/Linux)
```

---

## 6. Data licence and citation

The MIT-BIH Arrhythmia Database is distributed by PhysioNet under the Open Data Commons
Attribution License v1.0. When using it, cite:

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol Mag
  2001; 20(3): 45-50.
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet.
  Circulation 2000; 101(23): e215-e220.

## 7. Licence

Code: MIT License (see `LICENSE`). If you use this code, please cite the article (see
`CITATION.cff`).
