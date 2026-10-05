# Patient-independent ECG arrhythmia classification on MIT-BIH: a matched-protocol benchmark of a lightweight 1D-CNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproducible code for a **patient-independent (inter-patient)** benchmark of heartbeat
classification on the MIT-BIH Arrhythmia Database: a lightweight one-dimensional convolutional
neural network (1D-CNN, 9,412 parameters) evaluated with the AAMI classes N, S, V and F, compared
with class-imbalance strategies (class weighting, focal loss, oversampling), an RR-interval fusion
model, a 1D-ResNet and a random forest, and accompanied by a **matched comparison of patient-wise
and beat-wise cross-validation**, Grad-CAM explainability and an efficiency analysis — all with
five random seeds and patient-level bootstrap confidence intervals.

> **Aim.** To measure how much of the reported performance of ECG arrhythmia classifiers comes
> from the evaluation protocol rather than the model: the same data, model, tuning and training
> budget are evaluated with patient-wise and beat-wise cross-validation, and the protocol effect
> is separated from the model's lack of rhythm information by a 2 × 2 design (CNN with and
> without RR-interval features × patient-wise or beat-wise evaluation).

---

## ⚠️ Data availability and ethics (read first)

**This repository contains code only. It contains NO ECG recordings and NO derived beat-level data.**

The [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) is public and
de-identified and is distributed by PhysioNet under the Open Data Commons Attribution License
v1.0. No ethics approval or data-use agreement is required. The scripts download the 44 records
used here directly from PhysioNet (or import a manually downloaded copy):

- **Records:** the 44 non-paced records; paced records 102, 104, 107 and 217 are excluded
  (ANSI/AAMI EC57).
- **Split:** de Chazal inter-patient split, DS1 (22 records, training) and DS2 (22 records, test);
  listed in `splits/records_DS1_DS2.csv`.

Raw records, extracted beats, model predictions and trained weights are **excluded by design**
(see `.gitignore`).

---

## Repository structure

```
src/             Data preparation (Python): download of MIT-BIH from PhysioNet (01), import of a
                 manually downloaded copy (01b), beat extraction, 0.5-40 Hz band-pass filtering,
                 280-sample beat windows and AAMI labelling into DS1/DS2 (02).
src_rev/         Experiments and statistics (Python + PyTorch): configuration, data loading and
                 RR-interval features, models (1D-CNN, CNN+RR, 1D-ResNet), training, metrics,
                 the experiment orchestrator (run_revision.py) and the statistical analysis,
                 tables and figures (analyze_revision.py).
splits/          DS1/DS2 record lists.
requirements.txt Python dependencies.
CITATION.cff     Citation metadata.
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. A CUDA-enabled PyTorch build and an NVIDIA GPU are recommended for the full run
(about 1–2 h on a desktop GPU); the quick check (`--smoke`) runs on a CPU. For a CUDA build of
PyTorch use the selector at [pytorch.org](https://pytorch.org/get-started/locally/).

## Reproducing the analyses

Run the steps in this order from the repository root:

```bash
# 1) download the 44 MIT-BIH records into data/raw/
python src/01_download_data.py
#    if physionet.org cannot be reached, download the ZIP from PhysioNet and import it:
#    python src/01b_import_local.py "path/to/mit-bih-arrhythmia-database-1.0.0.zip"

# 2) beat extraction and AAMI labelling -> data/processed/train.npz (DS1), test.npz (DS2)
python src/02_preprocess.py

# 3) all experiments -> results_rev/   (resumable: finished runs are kept and skipped)
python src_rev/run_revision.py --smoke     # optional end-to-end check (a few minutes)
python src_rev/run_revision.py

# 4) statistics, tables and figures -> results_rev/analysis/
python src_rev/analyze_revision.py
```

Individual stages can be run with `--stages` (`data, tune, ds, cv, gradcam, eff, summary`), e.g.
`python src_rev/run_revision.py --stages tune,ds`. The run log is written to
`results_rev/revision_log.txt`; tables are written as CSV files to `results_rev/analysis/tables/`,
figures (300 dpi) to `results_rev/analysis/figures/`, and every statistic used in the article to
`results_rev/analysis/summary.json`. GPU training is not bit-wise deterministic across hardware
and library versions, so small differences in the third decimal are expected.

## Leakage controls (design summary)

- **Patient-level partitioning** — DS1 and DS2 share no patient; in the patient-wise
  cross-validation all beats of a record are kept in the same fold (`StratifiedGroupKFold`,
  group = record).
- **Tuning inside the training data only** — learning rate, batch size and weight decay are chosen
  by an inner patient-wise 3-fold cross-validation on DS1; DS2 is never used for any choice.
- **Matched protocol comparison** — patient-wise and beat-wise 5-fold cross-validation use the same
  44 records, model, hyper-parameters, number of epochs and seeds, and models are retrained in
  every fold; only the partitioning rule differs.
- **Training-only statistics** — class weights and the standardisation of RR features are computed
  on the training beats of each split or fold.
- **Patient-level uncertainty** — 95% confidence intervals and paired tests use a record-level
  bootstrap (2,000 resamples of patients), so that correlated beats of one patient are never
  treated as independent observations.
- **Explicit metric definitions** — macro-F1 is the unweighted mean of the per-class F1 of N, S, V
  and F; PPV is 0 for a class with no predicted beats and F1 is 0 when Se + PPV = 0.

## Citation

> Dündar, M. S. (2026). *Fair evaluation matters: an inter-patient benchmark of a lightweight 1D
> convolutional neural network for ECG arrhythmia classification on the MIT-BIH database.*
> Manuscript under review.

*(Will be updated with DOI/journal once available.)*

## License

Code is released under the [MIT License](LICENSE). This license covers **the code only**; the
MIT-BIH Arrhythmia Database remains under its PhysioNet licence (ODC Attribution License v1.0) and
is not redistributed here.

## Author

**Mehmet Sait Dündar**, Medical Imaging Techniques, Halil Bayraktar Health Services Vocational School, Erciyes University, Kayseri, Türkiye.
ORCID: [0000-0002-0336-4825](https://orcid.org/0000-0002-0336-4825).

## Acknowledgement

Data were obtained from the MIT-BIH Arrhythmia Database on PhysioNet. Users of the data should cite
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng Med Biol Mag*
2001; 20(3): 45-50, and Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation* 2000; 101(23): e215-e220.
