"""
Central configuration for the ECG arrhythmia study.

Everything that controls the experiment lives here so that the whole pipeline
is reproducible from a single place. You normally do NOT need to edit anything
except, optionally, the hyper-parameters near the bottom.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (everything is created automatically relative to the project root)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"          # downloaded MIT-BIH records
DATA_PROC = PROJECT_ROOT / "data" / "processed"   # cached beats (train/test .npz)
RESULTS = PROJECT_ROOT / "results"                # metrics, tables, figures, models
for _p in (DATA_RAW, DATA_PROC, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Dataset constants (MIT-BIH Arrhythmia Database, PhysioNet 'mitdb')
# ----------------------------------------------------------------------------
PN_DIR = "mitdb"        # PhysioNet database slug used by wfdb
FS = 360                # sampling frequency (Hz)
PREFERRED_LEAD = "MLII" # use this lead when present, else fall back to lead 0

# Beat window around the annotated R-peak (samples). 140+140 = 280 ~ 0.78 s.
WIN_LEFT = 140
WIN_RIGHT = 140
BEAT_LEN = WIN_LEFT + WIN_RIGHT

# Band-pass filter for baseline-wander / high-frequency noise removal.
BP_LOW = 0.5   # Hz
BP_HIGH = 40.0 # Hz
BP_ORDER = 3

# ----------------------------------------------------------------------------
# AAMI EC57 class scheme. Five classes: N, S(VEB), V(EB), F, Q.
# Maps every MIT-BIH beat annotation symbol to one AAMI super-class.
# Symbols not in this map (non-beat annotations) are ignored.
# ----------------------------------------------------------------------------
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

AAMI = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic beat (SVEB)
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic beat (VEB)
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced
    "/": "Q", "f": "Q", "Q": "Q",
}

# ----------------------------------------------------------------------------
# Inter-patient split (de Chazal et al., 2004). No patient overlap between sets.
# Paced records 102, 104, 107, 217 are excluded per AAMI and appear in neither.
# ----------------------------------------------------------------------------
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]   # training
DS2 = [100, 103, 105, 111, 113, 117, 121, 123,
       200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]  # test
ALL_RECORDS = DS1 + DS2

# ----------------------------------------------------------------------------
# Model / training hyper-parameters (lightweight 1D-CNN)
# ----------------------------------------------------------------------------
SEED = 42
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1      # fraction of DS1 records held out for early-stopping
FOCAL_GAMMA = 2.0       # used when --imbalance focal
N_FOLDS_INTRA = 5       # stratified folds for the intra-patient (optimistic) protocol
