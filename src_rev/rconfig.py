"""
Configuration of the experiments. Everything that defines the experimental design lives here.

Key design choices:
  * four AAMI classes (N, S, V, F); the Q class (15 beats) is excluded
  * hyper-parameters tuned by inner patient-wise cross-validation on DS1 only
  * five random seeds for every configuration (mean +/- SD)
  * matched protocol comparison: patient-wise vs beat-wise 5-fold CV on the same data
  * RR-interval fusion model (separates model-design effect from protocol effect)
  * deeper 1D-ResNet comparator, focal-loss gamma grid, class-balanced oversampling
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = ROOT / "data" / "processed"
OUT = ROOT / "results_rev"

# ---------------------------------------------------------------- data
CLASSES = ["N", "S", "V", "F"]          # AAMI classes analysed (Q excluded)
K = len(CLASSES)
FS = 360
BEAT_LEN = 280
R_IDX = 140                             # R-peak position inside the beat window
RR_LOCAL_WINDOW = 10                    # beats used for the causal local-average RR

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123,
       200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# Grad-CAM analysis regions (sample indices inside the 280-sample window, R at 140)
REGIONS = {
    "pre-beat (TP)": (0, 68),       # -389 to -200 ms
    "P wave / PR":   (68, 122),     # -200 to  -50 ms
    "QRS":           (122, 159),    #  -50 to  +50 ms
    "ST-T":          (159, 280),    #  +50 to +389 ms
}

# ---------------------------------------------------------------- experiment design
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS_CV = 5                          # matched patient-wise vs beat-wise CV
INNER_FOLDS = 3                         # inner patient-wise CV on DS1 for tuning
MAX_EPOCHS_TUNE = 60
PATIENCE = 10                           # early stopping on inner-validation macro-F1
MIN_EPOCHS = 5

GRID_CNN = [dict(lr=lr, bs=bs, wd=wd)
            for lr in (3e-3, 1e-3, 3e-4) for bs in (128, 256) for wd in (0.0, 1e-4)]
GRID_RESNET = [dict(lr=lr, bs=bs, wd=1e-4) for lr in (1e-3, 3e-4) for bs in (128, 256)]
FOCAL_GAMMAS = [0.5, 1.0, 2.0, 3.0, 5.0]
RF_TREES = 200


def apply_smoke():
    """Tiny settings used only to check that every stage runs end-to-end."""
    global OUT, SEEDS, MAX_EPOCHS_TUNE, PATIENCE, MIN_EPOCHS, GRID_CNN, GRID_RESNET
    global FOCAL_GAMMAS, RF_TREES
    import os
    OUT = Path(os.environ.get("REV_SMOKE_OUT", str(ROOT / "results_rev_smoke")))
    SEEDS = [0, 1]
    MAX_EPOCHS_TUNE = 2
    PATIENCE = 1
    MIN_EPOCHS = 1
    GRID_CNN = GRID_CNN[:2]
    GRID_RESNET = GRID_RESNET[:1]
    FOCAL_GAMMAS = [1.0, 2.0]
    RF_TREES = 20
