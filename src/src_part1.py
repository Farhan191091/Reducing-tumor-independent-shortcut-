# %% [markdown]
# # Do brain-tumour classifiers look at the tumour? — Run 6 (FINAL notebook)
# **Main task:** 3-class tumour typing (glioma / meningioma / pituitary) on Figshare T1c slices, patient-level split.
#
# **Story of the paper (Runs 1-5)**
# 1. **Problem:** standard models classify Figshare tumours *without needing the tumour* (pituitary 97-99% correct with
#    the tumour hidden); public "external" datasets are mostly copies of Figshare.
# 2. **Existing fixes fail:** multi-task segmentation, RRR, CAM guidance, counterfactual training, foundation models.
# 3. **Solutions that work (Run 5):** ROI models (classify only the found tumour region) and *tumour transplant*
#    training for an ordinary whole-slice model; skull control, shortcut alarm and metric checks passed.
#
# **What Run 6 adds (final)**
# | # | Addition | Why |
# |---|---|---|
# | 1 | **Bug fix:** an ROI model that finds no tumour gives an "unsure" answer; it is now counted as **abstained**, not as glioma | correct per-class numbers |
# | 2 | **ROI-m06-Transplant** = ROI crop with more context (margin 0.6) **+** tumour transplant | can we keep context *and* remove its shortcut? |
# | 3 | Two extra seeds of **ROI-Crop-m06** (best trade-off in Run 5) | stable numbers for the paper |
# | 4 | **High-resolution explanations** for every model incl. the ROI models (crop-level maps pasted back), blood-red colour scale | the main model's heat maps |
# | 5 | **Faithfulness test** (deletion curves) + **3D heat-map surfaces** + ROI sanity check | do the red areas really drive the decision? |
# | 6 | **Main results at a glance** figure + final success-criteria check | paper-ready summary |
#
# **How to run (final):** attach ALL artifact datasets (`mtk_planA_artifacts`, `mtk-run3-part2`, `mtk-run5-artifacts`,
# `mtk-run6-artifacts`) + the data inputs, click **Run All**. Every model is found -> no training, ~1.5 h of tables/figures.
# (Fresh start without artifacts: the same notebook trains everything that is missing.) Download the zips as soon as
# cell 3.6 shows the links (and again at the end, cell 12.3).

# %%
# ─── 0.1 Install (Kaggle already has torch / torchvision / timm) ───
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagehash"], check=False)

# %%
# ─── 0.2 Imports + configuration ───
import os, re, gc, json, time, math, random, hashlib, shutil, warnings, platform
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"      # hide the noisy download bars of pretrained weights
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import timm
from scipy import stats as sps
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm   # plain text bars (no 'Error displaying widget')
from multiprocessing import Pool
warnings.filterwarnings("ignore")

# ===================== RUN SWITCHES =====================
RUN_TRAINING   = True    # False = evaluation only (uses saved checkpoints)
RUN_LR_SEARCH  = True    # short LR search on fold 1 (done once, then remembered)
RUN_EVALUATION = True    # CV / test / external / calibration tables
RUN_XAI        = "auto"  # "auto" = as soon as the main models (seed 0) are trained; results are cached
RUN_RELIANCE   = True    # tumour-hiding test for every trained model (cached)
RUN_SHORTCUT   = True    # old 4-class shortcut case study (no training)

# FULL training plan, in priority order. You do NOT edit this between runs:
# each run trains whatever is not finished yet, then stops in time and makes all tables.
TRAIN_PLAN = [
    # --- already trained in Runs 1-3 (loaded from the attached artifacts, not retrained) ---
    ("MTK", [0, 1, 2]), ("CNN-Backbone", [0]), ("CNN-Backbone-Wide", [0]),
    ("ResNet-50", [0]), ("EfficientNet-B0", [0]), ("ConvNeXt-Tiny", [0]), ("DeiT-Small", [0]),
    ("MTK-noSeq", [0]), ("MTK-noKAN", [0]), ("MTK-noCNN", [0]),
    ("CNX-Plain", [0]), ("CTR-Net", [0]), ("CTR-noCF", [0]), ("CTR-noSeg", [0]),
    ("MTL-Seg", [0]), ("RRR", [0]), ("CAM-Guide", [0]),
    ("CTR-Net", [1, 2]), ("CNX-Plain", [1, 2]), ("CTR-MambaKAN", [0]),
    # --- Run 4 remedy (its checkpoints could not be downloaded -> retrained in Run 5) ---
    ("ROI-Crop", [0]),
    # --- NEW in Run 5: the solutions (seed 0) ---
    ("ROI-Loc", [0]),            # 1: appearance + legitimate location
    ("CNX-Transplant", [0]),     # 2: training-time fix for an ordinary whole-slice model
    ("CNX-Strip", [0]),          # 3: skull control
    ("ROI-Crop-m15", [0]),       # 1b: context sweep (margin 1.5)
    ("ROI-Crop-m06", [0]),       # 1b: context sweep (margin 0.6)
    ("ROI-Mask", [0]), ("ROI-Loc", [1, 2]), ("ROI-Crop", [1, 2]), ("CNX-Transplant", [1, 2]),
    # --- NEW in Run 6 (final) ---
    ("ROI-m06-Transplant", [0]),   # context (margin 0.6) + tumour transplant
    ("ROI-Crop-m06", [1, 2]),      # extra seeds of the best trade-off model
    ("ROI-m06-Transplant", [1, 2]),# extra seeds if time allows (otherwise reported with seed 0 only)
]
TRAIN_QUEUE = TRAIN_PLAN        # trained in this order; whatever does not fit is left for the next run
MAX_TRAIN_HOURS = 6.5    # stop STARTING new training jobs after this (evaluation + new XAI need ~3 h)
SESSION_START = time.time()   # clock starts when the notebook starts

# ===================== PATHS =====================
KAGGLE_IN = Path("/kaggle/input")
FIGSHARE_DIR   = KAGGLE_IN / "datasets/raufuumme/brain-tumor-patient-organized/brain_tumor_patient_organized"
EXT_DIR_BRISC  = KAGGLE_IN / "datasets/farhan1910/brisc-images/brisc_images"
BRISC_MASK_DIR = KAGGLE_IN / "datasets/farhan1910/brisc25-masks/brisc25_masks"
EXT_DIR_MENDELEY = KAGGLE_IN / "datasets/mdkayumhossain/mendeley-dataset/Mendeley_N/Mendeley_N"
OLD_ARTIFACT_DIR = KAGGLE_IN / "datasets/mdkayumhossain/barain-all-saved-models"   # old 4-class run
BRATS_DIR = KAGGLE_IN / "brats-2d-slices"          # folder that contains manifest.csv
BRATS_MIN_TC_PIXELS = 100
FIGSHARE_MAT_DIR = KAGGLE_IN / "datasets/farhan1910/figshare-original-mat"   # original .mat files (tumour masks)
BDNEURO_DIR = KAGGLE_IN / "datasets/farhan1910/bdneuro-mri"                   # Bangladeshi dataset (mostly copies)
BRATS_FREE_DIR = KAGGLE_IN / "datasets/farhan1910/brats-free-slices"          # tumour-free BraTS slices
MSOUD_DIR = KAGGLE_IN / "datasets/masoudnickparvar/brain-tumor-mri-dataset"   # popular Kaggle dataset (audit only)
# Viewing plane of each cluster of cell 1.9 (read from the Run 4 sheet). Clusters are matched by their SIZE, so a
# different cluster numbering cannot mix them up; if the sizes differ, the plane analysis is skipped with a warning.
PLANE_BY_SIZE = {165: "axial", 321: "coronal", 438: "sagittal", 253: "sagittal", 320: "coronal", 317: "axial",
                 228: "sagittal", 172: "axial", 138: "mixed", 118: "mixed", 365: "coronal", 229: "axial"}
ARTIFACT_INPUT_DIRS = []   # leave empty: the previous run's output is found automatically

WORK = Path("/kaggle/working")
MODEL_DIR, OUT_DIR, FIG_DIR = WORK / "models", WORK / "outputs", WORK / "figures"
for d in [MODEL_DIR, OUT_DIR, FIG_DIR, OUT_DIR / "runs", OUT_DIR / "oof", OUT_DIR / "preds", OUT_DIR / "tables", OUT_DIR / "xai"]:
    d.mkdir(parents=True, exist_ok=True)

def locate(path, marker=None):
    """Return `path` if it exists, otherwise search /kaggle/input for a folder
    with the same name (and containing `marker` if given)."""
    path = Path(path)
    if path.exists():
        return path
    if KAGGLE_IN.exists():
        for cand in KAGGLE_IN.rglob(path.name):
            if cand.is_dir() and (marker is None or any(cand.rglob(marker))):
                print(f"  auto-located {path.name}: {cand}")
                return cand
    return path

FIGSHARE_DIR = locate(FIGSHARE_DIR, "*.png")
if not BDNEURO_DIR.exists() and KAGGLE_IN.exists():
    BDNEURO_DIR = next((d for d in KAGGLE_IN.glob("*/*/*") if d.is_dir() and "bdneuro" in d.name.lower()), BDNEURO_DIR)
MASK_NPZ_INPUT = next(iter(KAGGLE_IN.rglob("figshare_masks_224.npz")), None) if KAGGLE_IN.exists() else None
if not (BRATS_FREE_DIR / "manifest_free.csv").exists() and KAGGLE_IN.exists():
    BRATS_FREE_DIR = next((h.parent for h in KAGGLE_IN.rglob("manifest_free.csv")), BRATS_FREE_DIR)
if not (MSOUD_DIR / "Training").exists() and KAGGLE_IN.exists():
    MSOUD_DIR = next((d.parent for d in KAGGLE_IN.glob("**/Training") if (d / "glioma").exists() and (d.parent / "Testing").exists()), MSOUD_DIR)
EXT_DIR_BRISC = locate(EXT_DIR_BRISC)
BRISC_MASK_DIR = locate(BRISC_MASK_DIR)
EXT_DIR_MENDELEY = locate(EXT_DIR_MENDELEY)
OLD_ARTIFACT_DIR = locate(OLD_ARTIFACT_DIR)
if not any(FIGSHARE_MAT_DIR.glob("brainTumorDataPublic_*")) and KAGGLE_IN.exists():
    _hits = sorted({h.parent for h in KAGGLE_IN.glob("**/brainTumorDataPublic_*")})
    if _hits:
        FIGSHARE_MAT_DIR = _hits[0]
if not (BRATS_DIR / "manifest.csv").exists() and KAGGLE_IN.exists():
    hits = list(KAGGLE_IN.rglob("manifest.csv"))
    hits = [h.parent for h in hits if (h.parent / "gli").exists() or (h.parent / "men").exists()]
    if hits:
        BRATS_DIR = hits[0]

# ===================== EXPERIMENT SETTINGS =====================
CLASSES = ["glioma", "meningioma", "pituitary"]
N_CLASSES = len(CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
IDX2CLASS[-1] = "unsure (abstained)"   # ROI model found no tumour
CLASSES_4 = CLASSES + ["notumor"]           # only for the shortcut study
PREFIX = {"glioma": "g", "meningioma": "m", "pituitary": "p", "notumor": "n"}

IMG_SIZE, BATCH_SIZE = 224, 32
NUM_EPOCHS, PATIENCE, WARMUP_EPOCHS = 80, 15, 3
WEIGHT_DECAY, LABEL_SMOOTHING, MIX_PROB = 1e-2, 0.10, 0.5
N_FOLDS, SPLIT_SEED, TEST_FRACTION = 5, 42, 0.15
DEFAULT_LR = {"scratch": 1e-4, "pretrained": 5e-5}
LR_GRID = {"scratch": [3e-4, 1e-4], "pretrained": [1e-4, 5e-5, 1e-5]}
# LR search is run once per model FAMILY; ablations use the LR of their parent model
LR_SEARCH_MODELS = ["MTK", "CNN-Backbone", "ResNet-50", "EfficientNet-B0", "ConvNeXt-Tiny", "DeiT-Small"]
LR_INHERIT = {"MTK-noSeq": "MTK", "MTK-noKAN": "MTK", "MTK-noCNN": "MTK", "CNN-Backbone-Wide": "CNN-Backbone",
              # TA-MTK = ConvNeXt-Tiny backbone + new heads: backbone uses ConvNeXt-Tiny's LR, new heads 10x
              "TA-MTK": "ConvNeXt-Tiny", "TA-MTK-noSeg": "ConvNeXt-Tiny",
              "TA-MTK-noSeq": "ConvNeXt-Tiny", "TA-MTK-noKAN": "ConvNeXt-Tiny",
              "CNX-Plain": "ConvNeXt-Tiny", "CTR-Net": "ConvNeXt-Tiny", "CTR-noCF": "ConvNeXt-Tiny",
              "CTR-noSeg": "ConvNeXt-Tiny", "MTL-Seg": "ConvNeXt-Tiny", "RRR": "ConvNeXt-Tiny",
              "CAM-Guide": "ConvNeXt-Tiny", "CTR-MambaKAN": "ConvNeXt-Tiny",
              "ROI-Crop": "ConvNeXt-Tiny", "ROI-Mask": "ConvNeXt-Tiny",
              "ROI-Loc": "ConvNeXt-Tiny", "ROI-Crop-m06": "ConvNeXt-Tiny", "ROI-Crop-m15": "ConvNeXt-Tiny",
              "CNX-Transplant": "ConvNeXt-Tiny", "CNX-Strip": "ConvNeXt-Tiny", "ROI-m06-Transplant": "ConvNeXt-Tiny"}
ROI_SEG_SOURCE = "MTL-Seg"   # its tumour map (same fold) decides where ROI models look
ROI_MARGIN = 0.2             # box around the found tumour, +20% on each side (some context)
ROI_MIN_PIXELS = 20          # smaller "tumour" = nothing found -> the ROI model says "unsure" (uniform)
CF_HIDE_WEIGHT = 0.5       # counterfactual rule 1: tumour hidden -> prediction must be uniform (unsure)
CF_KEEP_WEIGHT = 0.5       # counterfactual rule 2: random spot hidden -> same label as the original
GUIDE_WEIGHT = 1.0         # RRR / CAM-Guide: weight of the "evidence outside the tumour" penalty
GUIDE_BATCH = 8            # images per batch used for the (double-backprop) guidance penalties
HEAD_LR_MULT = 10          # new (randomly initialised) heads of TA-MTK learn 10x faster than the pretrained backbone
SEG_LOSS_WEIGHT = 0.5      # loss = classification + 0.5 x (BCE + Dice) on the tumour mask (fixed, not tuned)
# --- Run 5 solution settings (fixed before the run, not tuned) ---
TRANSPLANT_PROB = 0.5      # share of training images that receive a tumour from a patient of ANOTHER class
STRIP_ERODE = 17           # skull stripping = head mask eroded by a 17x17 kernel (same "interior" as the skull test)
ALARM_DROP = 0.20          # shortcut alarm: answer only if hiding the found tumour lowers the confidence by >= 0.20
METRIC_SIZES = [9, 25, 41] # hidden-area sizes for the metric check (9 = the main test; kernel size in pixels)
NUM_WORKERS = min(4, os.cpu_count() or 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_DP_FOR_PRETRAINED = False   # one GPU for every model (DataParallel leaked RAM and was 15x slower)
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_BOOT = 2000
CONF_THRESHOLD = 0.85

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
from matplotlib.colors import LinearSegmentedColormap
# Blood-red heat-map colour scale (transparent -> dark red -> bright red), SAME scale (0-1) for every model
BLOOD = LinearSegmentedColormap.from_list("blood", [(0.0, (0.35, 0, 0, 0)), (0.2, (0.4, 0, 0, 0)), (0.4, (0.6, 0, 0, 0.45)),
                                                    (0.7, (0.85, 0, 0, 0.78)), (1.0, (1.0, 0.05, 0.05, 0.93))])
BLOOD_SOLID = LinearSegmentedColormap.from_list("blood_solid", ["#1a0000", "#5c0000", "#a30000", "#e00000", "#ff3b3b"])
CLASS_COLORS = {"glioma": "#d1495b", "meningioma": "#00798c", "pituitary": "#edae49", "notumor": "#30638e"}

print("Torch", torch.__version__, "| timm", timm.__version__, "| GPUs:", torch.cuda.device_count())
for name, p in [("Figshare", FIGSHARE_DIR), ("BRISC", EXT_DIR_BRISC), ("BRISC masks", BRISC_MASK_DIR),
                ("Mendeley", EXT_DIR_MENDELEY), ("Old 4-class artifacts", OLD_ARTIFACT_DIR), ("BraTS slices", BRATS_DIR),
                ("Figshare .mat (masks)", FIGSHARE_MAT_DIR), ("BDNeuro-MRI", BDNEURO_DIR),
                ("Mask file (optional)", MASK_NPZ_INPUT or Path("/none")),
                ("BraTS tumour-free slices", BRATS_FREE_DIR), ("Masoud Kaggle dataset", MSOUD_DIR)]:
    print(f"  {'OK ' if Path(p).exists() else 'MISSING'}  {name:22s} {p}")

# %%
# ─── 0.3 Utilities: seeds, hashing, json, artifact sync ───
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

def load_json(path, default=None):
    path = Path(path)
    return json.load(open(path)) if path.exists() else default

def banner(title, note=""):
    print("\n" + "=" * 70 + f"\n▶ {title}" + (f"\n  {note}" if note else "") + "\n" + "=" * 70, flush=True)

def ram_text():
    try:
        import psutil
        v = psutil.virtual_memory()
        return f"RAM {v.used / 1e9:.1f}/{v.total / 1e9:.0f} GB" + ("  ⚠ RAM HIGH" if v.percent > 85 else "")
    except Exception:
        return ""

ROLE = {"ROI-Crop": "REMEDY: classify only the tumour region found by the tumour map (box with context)",
        "ROI-Mask": "REMEDY: like ROI-Crop, but everything outside the found tumour is blacked out",
        "CTR-Net": "TESTED REMEDY (Run 3): tumour-mask head + mask-guided pooling + counterfactual rules",
        "CTR-noCF": "ABLATION: CTR-Net without the counterfactual rules",
        "CTR-noSeg": "ABLATION: counterfactual rules only (no mask head, no mask-guided pooling)",
        "CTR-MambaKAN": "ABLATION: CTR-Net + Mamba (Seq) + KAN pathways",
        "CNX-Plain": "BASELINE: ConvNeXt-T, same pipeline, no guidance",
        "MTL-Seg": "EXISTING METHOD: multi-task segmentation + classification",
        "RRR": "EXISTING METHOD: Right-for-the-Right-Reasons (input-gradient penalty outside the tumour)",
        "CAM-Guide": "EXISTING METHOD: class-activation-map penalty outside the tumour",
        "ROI-Loc": "SOLUTION 1: tumour appearance (crop) + tumour location relative to the brain (6 numbers)",
        "ROI-Crop-m06": "CONTEXT SWEEP: ROI-Crop with a wider box (margin 0.6)",
        "ROI-Crop-m15": "CONTEXT SWEEP: ROI-Crop with a much wider box (margin 1.5)",
        "CNX-Transplant": "SOLUTION 2: ordinary ConvNeXt trained with tumour transplants between classes",
        "CNX-Strip": "CONTROL: ordinary ConvNeXt on skull-stripped images (is the BraTS gain just 'no skull'?)",
        "ROI-m06-Transplant": "COMBINED: ROI crop with context (margin 0.6) + tumour transplant (context carries no label)"}

def role_of(name):
    if name in ROLE: return ROLE[name]
    if name == "TA-MTK": return "NEW PROPOSED model (tumour-aware: ConvNeXt-Tiny + Mamba + KAN + tumour mask head)"
    if name.startswith("TA-MTK-no"): return "ABLATION of the new proposed model"
    if name == "MTK": return "Original MTK (submitted version, fixed)"
    if name.startswith("MTK-no"): return "ABLATION (MTK with one pathway removed)"
    if name.startswith("CNN-Backbone"): return "ABLATION / baseline (plain CNN, no Mamba, no KAN)"
    return "BASELINE (ImageNet-pretrained)"

def hms(sec):
    sec = int(sec); return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"

def figshare_pid(path):
    m = re.match(r"^[gmpn]_(.+)_s_\d+$", Path(path).stem)
    if m is None:
        raise ValueError(f"Cannot parse patient id: {path}")
    return m.group(1)

ZIP_PREFIX = "mtk_run6"
SYNCED_CKPTS = set()        # checkpoints copied from attached inputs (already saved -> not zipped again)
ZIP_PART_GB = 1.8          # every zip stays below this size, so the browser download works

def make_artifact_zip(with_figures=True):
    """Small zips (each < 1.8 GB): one RESULTS zip (tables, records, predictions, figures) and
    MODELS part zips (checkpoints). Download all of them; attach all of them next run."""
    import zipfile
    for old in WORK.glob(f"{ZIP_PREFIX}_*.zip"):
        old.unlink()
    res = WORK / f"{ZIP_PREFIX}_results.zip"
    with zipfile.ZipFile(res, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT_DIR.rglob("*")):
            if f.is_file() and f.suffix in (".json", ".npy", ".npz", ".csv", ".md"):
                z.write(f, f"outputs/{f.relative_to(OUT_DIR)}")
        if with_figures:
            for f in sorted(FIG_DIR.glob("*.png")):
                z.write(f, f"figures/{f.name}")
    parts, cur, size = [], [], 0
    for f in sorted(p for p in MODEL_DIR.glob("*.pt") if p.name not in SYNCED_CKPTS):   # only NEW models
        s = f.stat().st_size
        if cur and size + s > ZIP_PART_GB * 1e9:
            parts.append(cur); cur, size = [], 0
        cur.append(f); size += s
    if cur:
        parts.append(cur)
    for k, files in enumerate(parts, 1):
        with zipfile.ZipFile(WORK / f"{ZIP_PREFIX}_models_part{k}.zip", "w", zipfile.ZIP_STORED) as z:
            for f in files:
                z.write(f, f"models/{f.name}")
    print("Saved (download ALL of these files from the Output panel -> /kaggle/working):")
    for f in sorted(WORK.glob(f"{ZIP_PREFIX}_*.zip")):
        print(f"   {f.name:32s} {f.stat().st_size / 1e9:5.2f} GB")
    print(f"   ({len([p for p in MODEL_DIR.glob('*.pt') if p.name not in SYNCED_CKPTS])} new model checkpoints; "
          f"{len(SYNCED_CKPTS)} older ones are already in your attached datasets - keep attaching those)")

CKPT_RE = re.compile(r"^.+_s\d+_f\d+\.pt$")
KEEP_NAMES = ("lr_choice.json", "fold_stats_3class.json", "overlap_audit.csv", "split_3class.csv",
              "figshare_masks_224.npz", "image_props.csv", "plane_clusters.csv")

def sync_artifacts():
    """Copy checkpoints and saved results of earlier runs (zips or folders anywhere in /kaggle/input)."""
    import zipfile
    roots = []
    if KAGGLE_IN.exists():
        for zp in [z for pat in ("mtk_planA_*.zip", "mtk_run3_*.zip", "mtk_run4_*.zip", "mtk_run5_*.zip", "mtk_run6_*.zip") for z in KAGGLE_IN.rglob(pat)]:
            dst = (Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp")) / "prev_artifacts" / zp.stem
            if not dst.exists():
                dst.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(zp).extractall(dst)
            roots.append(dst)
        roots.append(KAGGLE_IN)
    n = 0
    for root in roots:
        for src in root.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix == ".pt" and CKPT_RE.match(src.name):
                dst = MODEL_DIR / src.name; SYNCED_CKPTS.add(src.name)
            elif src.parent.name in ("runs", "oof", "preds", "xai", "reliance", "reliance2", "alarm", "fm") and src.parent.parent.name == "outputs":
                dst = OUT_DIR / src.parent.name / src.name
            elif src.name in KEEP_NAMES:
                dst = OUT_DIR / src.name
            else:
                continue
            if dst.exists() and (dst.is_symlink() or dst.stat().st_size > 0):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()                              # empty file from a full disk -> replace it
            if src.suffix == ".pt":
                dst.symlink_to(src)                       # LINK old models (saves ~10 GB; Kaggle allows ~20 GB)
            else:
                shutil.copy2(src, dst)
            n += 1
    print(f"Earlier results copied/linked: {n} files | model checkpoints available: {len(list(MODEL_DIR.glob('*.pt')))} | "
          f"disk free {shutil.disk_usage(WORK).free / 1e9:.1f} GB")

set_seed(SPLIT_SEED)
sync_artifacts()

# %% [markdown]
# ## 0.4 Success criteria (written down BEFORE the results exist)
# These thresholds were fixed before Run 5. Section 12 checks every one automatically and prints ✔ / ✘.
# A ✘ is still a valid, reportable result - the criteria exist so that we cannot move the goal posts afterwards.

# %%
# ─── 0.4 Pre-registered success criteria ───
CRITERIA = [
    ("S1a", "ROI-Loc", "CV macro-F1 >= 96.0% (wins back the accuracy ROI-Crop lost)"),
    ("S1b", "ROI-Loc", "tumour reliance >= 0.60 AND skull/position-only <= 36% (shortcut stays removed)"),
    ("S1c", "ROI-Loc", "BraTS glioma-vs-meningioma AUC >= ROI-Crop - 0.01 (generalisation kept)"),
    ("S2a", "CNX-Transplant", "tumour reliance >= 0.50 (CNX-Plain: 0.35) with the swap test"),
    ("S2b", "CNX-Transplant", "tumour reliance >= 0.45 with the SECOND hiding method (tissue from another patient)"),
    ("S2c", "CNX-Transplant", "CV macro-F1 >= 95.0% and pituitary still-correct-when-hidden < 80% (CNX-Plain: 98.7%)"),
    ("S3",  "CNX-Strip", "if BraTS AUC of CNX-Strip < ROI-Crop - 0.02 -> ROI's BraTS gain is NOT explained by the skull"),
    ("S4a", "Shortcut alarm", "on BraTS, alarm-based ranking has lower AURC than confidence alone (CNX-Plain)"),
    ("S4b", "Shortcut alarm", "separates tumour slices from tumour-free slices with AUROC >= 0.75 (CNX-Plain)"),
    ("S5",  "Metric check", "model ranking by reliance is stable: Spearman(swap, other-patient) >= 0.80"),
    # --- added before Run 6 ---
    ("S6a", "ROI-m06-Transplant", "CV macro-F1 >= 96.0% (context brings the accuracy back)"),
    ("S6b", "ROI-m06-Transplant", "tumour reliance >= 0.60 (the extra context does not become a shortcut)"),
    ("S6c", "ROI-m06-Transplant", "BraTS AUC >= ROI-Crop - 0.01"),
    ("S7",  "XAI (ROI-Loc)", "share of heat inside the tumour >= 2x what a random map gets inside the same crop"),
    ("S8",  "XAI faithfulness", "deleting ROI-Loc's red pixels lowers its confidence faster than deleting random pixels"),
]
print(pd.DataFrame(CRITERIA, columns=["ID", "Experiment", "Success if ..."]).to_string(index=False))

# %% [markdown]
# ## 1. Data
# The split reproduces the **original** patient-level split exactly (same seed, same 4-class stratification),
# then removes the IXI no-tumour patients. Test patients and fold membership of every tumour patient are therefore
# identical to the submitted paper.

# %%
# ─── 1.1 Reproduce the original split, then keep the 3 tumour classes ───
records = []
for cls in CLASSES_4:
    folder = FIGSHARE_DIR / cls
    files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")))
    for f in files:
        m = re.match(rf"^{PREFIX[cls]}_(.+)_s_\d+$", f.stem)
        records.append({"filepath": str(f), "label4": cls, "label4_idx": CLASSES_4.index(cls),
                        "pid": m.group(1) if m else f.stem})
df_all = pd.DataFrame(records)
assert len(df_all), "No images found: check FIGSHARE_DIR"

df_pat = df_all.groupby("pid").agg(label4=("label4", "first"), label4_idx=("label4_idx", "first")).reset_index()
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
tv_idx, te_idx = next(sss.split(df_pat["pid"].values, df_pat["label4_idx"].values))
df_pat_tv = df_pat.iloc[tv_idx].reset_index(drop=True)
test_pids = set(df_pat.iloc[te_idx]["pid"])
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
val_fold_of = {}
for k, (_, vl) in enumerate(skf.split(df_pat_tv["pid"].values, df_pat_tv["label4_idx"].values), start=1):
    for p in df_pat_tv["pid"].values[vl]:
        val_fold_of[p] = k
df_all["role"] = np.where(df_all["pid"].isin(test_pids), "test", "dev")
df_all["val_fold"] = df_all["pid"].map(val_fold_of)

print(f"4-class reproduction: {len(df_all)} images, {df_all.pid.nunique()} patients, "
      f"test {(df_all.role=='test').sum()} images / {len(test_pids)} patients (original: 713 / 53)")
old_test_csv = OLD_ARTIFACT_DIR / "outputs" / "test_set_index.csv"
if old_test_csv.exists():
    old_pids = set(pd.read_csv(old_test_csv)["pid"].astype(str))
    print("  identical to the original test patients:", old_pids == {str(p) for p in test_pids})

# 3-class (main task)
df3 = df_all[df_all.label4 != "notumor"].copy()
df3["label"] = df3["label4_idx"].astype(int)          # glioma 0, meningioma 1, pituitary 2
assert set(df3.label) == {0, 1, 2}
FOLDS = {}
for k in range(1, N_FOLDS + 1):
    dev = df3[df3.role == "dev"]
    FOLDS[k] = {"train": list(zip(dev[dev.val_fold != k].filepath, dev[dev.val_fold != k].label)),
                "val":   list(zip(dev[dev.val_fold == k].filepath, dev[dev.val_fold == k].label))}
    tr_p = set(dev[dev.val_fold != k].pid); vl_p = set(dev[dev.val_fold == k].pid)
    assert not (tr_p & vl_p) and not (tr_p & test_pids) and not (vl_p & test_pids), "patient leakage"
TEST_ITEMS = list(zip(df3[df3.role == "test"].filepath, df3[df3.role == "test"].label))

df3.assign(split=np.where(df3.role == "test", "test", "fold" + df3.val_fold.fillna(0).astype(int).astype(str))) \
   [["filepath", "label4", "label", "pid", "split"]].to_csv(OUT_DIR / "split_3class.csv", index=False)
summary = df3.groupby(["role", "label4"]).agg(images=("filepath", "size"), patients=("pid", "nunique"))
print(summary)
for k in FOLDS:
    print(f"  fold {k}: train {len(FOLDS[k]['train'])} / val {len(FOLDS[k]['val'])} images")
print(f"  TEST: {len(TEST_ITEMS)} images, {df3[df3.role=='test'].pid.nunique()} patients")

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for a, role, title in [(ax[0], "dev", "Development (5-fold CV)"), (ax[1], "test", "Held-out test")]:
    d = df3[df3.role == role].groupby("label4").agg(images=("filepath", "size"), patients=("pid", "nunique")).reindex(CLASSES)
    bars = a.bar(CLASSES, d.images, color=[CLASS_COLORS[c] for c in CLASSES])
    for b, n_p in zip(bars, d.patients):
        a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n_p} pts", ha="center", va="bottom", fontsize=8)
    a.set(title=title, ylabel="Images")
fig.tight_layout(); fig.savefig(FIG_DIR / "fig_dataset.png"); plt.show()

# %%
# ─── 1.2 Image properties per source (also used in the shortcut study) ───
def native_props(path):
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode, Path(path).suffix.lower(), os.path.getsize(path)

props_csv = OUT_DIR / "image_props.csv"
if props_csv.exists() and len(pd.read_csv(props_csv)) == len(df_all):
    props = pd.read_csv(props_csv)                       # cached from an earlier run (reading headers takes ~10 min)
else:
    props = pd.DataFrame([dict(zip(["width", "height", "mode", "ext", "bytes"], native_props(p)), label4=l)
                          for p, l in tqdm(list(zip(df_all.filepath, df_all.label4)), desc="Reading image headers")])
    props.to_csv(props_csv, index=False)
prop_table = props.groupby("label4").agg(n=("width", "size"), width=("width", "median"), height=("height", "median"),
                                         ext=("ext", lambda s: ",".join(sorted(set(s)))),
                                         kB=("bytes", lambda s: round(s.median() / 1024, 1)))
print(prop_table)
prop_table.to_csv(OUT_DIR / "tables" / "image_properties_by_class.csv")

# %%
# ─── 1.3 External datasets: lists (tumour classes only) ───
ALIASES = {"glioma": "glioma", "glioma_tumor": "glioma", "meningioma": "meningioma", "meningioma_tumor": "meningioma",
           "pituitary": "pituitary", "pituitary_tumor": "pituitary", "gliomas": "glioma", "meningiomas": "meningioma",
           "pituitary_tumour": "pituitary", "glioma_tumour": "glioma", "meningioma_tumour": "meningioma"}

def list_external(root):
    items = []
    root = Path(root)
    if not root.exists():
        return items
    for folder in sorted(root.iterdir()):
        cls = ALIASES.get(folder.name.lower().replace(" ", "_").replace("-", "_"))
        if folder.is_dir() and cls:
            for f in sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))):
                items.append((str(f), CLASS2IDX[cls]))
    return items

def list_external_recursive(root):
    """Any sub-folder whose name is a tumour class (any depth, any case, .png/.jpg/.jpeg)."""
    items, root = [], Path(root)
    if not root.exists():
        return items
    for folder in sorted(d for d in root.rglob("*") if d.is_dir()):
        cls = ALIASES.get(folder.name.lower().strip().replace(" ", "_").replace("-", "_"))
        if cls:
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    items.append((str(f), CLASS2IDX[cls]))
    return items

BRISC_ITEMS = list_external(EXT_DIR_BRISC)
BD_ITEMS = list_external_recursive(BDNEURO_DIR)
MSOUD_TRAIN = list_external_recursive(MSOUD_DIR / "Training") if (MSOUD_DIR / "Training").exists() else []
MSOUD_TEST = list_external_recursive(MSOUD_DIR / "Testing") if (MSOUD_DIR / "Testing").exists() else []
BFREE_ITEMS, BFREE_META = [], pd.DataFrame()
if (BRATS_FREE_DIR / "manifest_free.csv").exists():
    BFREE_META = pd.read_csv(BRATS_FREE_DIR / "manifest_free.csv")
    BFREE_META["path"] = [str(BRATS_FREE_DIR / f) for f in BFREE_META["file"]]
    BFREE_META = BFREE_META[BFREE_META.path.map(lambda q: Path(q).exists())].reset_index(drop=True)
    BFREE_ITEMS = list(zip(BFREE_META.path, BFREE_META.patient_tumour_type.map(CLASS2IDX)))   # label = patient's tumour type (reference only)
MEND_ITEMS = list_external(EXT_DIR_MENDELEY)

BRATS_ITEMS, BRATS_META = [], pd.DataFrame()
if (BRATS_DIR / "manifest.csv").exists():
    BRATS_META = pd.read_csv(BRATS_DIR / "manifest.csv")
    BRATS_META["path"] = [str(BRATS_DIR / f) for f in BRATS_META["file"]]
    BRATS_META["y"] = BRATS_META["label"].map(CLASS2IDX)
    BRATS_META = BRATS_META[BRATS_META.path.map(lambda p: Path(p).exists())]
    # drop slices with a tiny tumour core (< 100 pixels of 240x240): not a fair "tumour visible" test
    BRATS_META = BRATS_META[BRATS_META.tc_pixels >= BRATS_MIN_TC_PIXELS].reset_index(drop=True)
    BRATS_ITEMS = list(zip(BRATS_META.path, BRATS_META.y))

BRISC_MASKS = {}
if BRISC_MASK_DIR.exists():
    for m in BRISC_MASK_DIR.rglob("*"):
        if m.suffix.lower() in (".png", ".jpg", ".jpeg"):
            BRISC_MASKS[m.stem.replace("_mask", "")] = str(m)

print(f"BraTS tumour-free slices: {len(BFREE_ITEMS)} (from {BFREE_META.patient_id.nunique() if len(BFREE_META) else 0} patients)")
for name, items in [("BRISC", BRISC_ITEMS), ("Mendeley", MEND_ITEMS), ("BraTS", BRATS_ITEMS), ("BDNeuro", BD_ITEMS),
                    ("Masoud-train", MSOUD_TRAIN), ("Masoud-test", MSOUD_TEST)]:
    c = Counter(IDX2CLASS[y] for _, y in items)
    print(f"{name:9s}: {len(items):5d} tumour images {dict(c)}")
print(f"BRISC masks indexed: {len(BRISC_MASKS)}")

# %%
# ─── 1.4 Duplicate / overlap audit (Reviewer 2) ───
# Perceptual hash (flip-invariant) finds candidates (Hamming <= 40 of 256 bits);
# a 64x64 pixel correlation (flip-invariant) confirms them: duplicate = correlation >= 0.95.
# Pairs with 0.85 <= r < 0.95 are shown for a manual check (adjacent slices / crops).
import imagehash

HAM_CAND, CORR_THR = 40, 0.95

def fingerprint(path):
    g = Image.open(path).convert("L")
    h1 = np.packbits(imagehash.phash(g.resize((256, 256)), hash_size=16).hash.ravel())
    h2 = np.packbits(imagehash.phash(g.transpose(Image.FLIP_LEFT_RIGHT).resize((256, 256)), hash_size=16).hash.ravel())
    small = np.asarray(g.resize((64, 64), Image.BILINEAR), np.float32).ravel()
    small = (small - small.mean()) / (small.std() + 1e-6)
    return h1, h2, small

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)

def fingerprints(items, cache_name):
    path = OUT_DIR / "preds" / f"fp_{cache_name}.npz"
    if path.exists():
        z = np.load(path)
        if len(z["h"]) == len(items):
            print(f"  {cache_name}: fingerprints loaded from cache ({len(items)} images)")
            return z["h"], z["hf"], z["s"]
    try:
        with Pool(max(1, os.cpu_count() or 1)) as pool:      # all CPU cores
            fps = list(tqdm(pool.imap(fingerprint, [p for p, _ in items], chunksize=32), total=len(items),
                            desc=f"  Fingerprinting {cache_name}"))
    except Exception as e:                                   # fallback: one core
        print("  parallel failed, using one core:", e)
        fps = [fingerprint(p) for p, _ in tqdm(items, desc=f"  Fingerprinting {cache_name}")]
    h = np.stack([a for a, _, _ in fps]); hf = np.stack([b for _, b, _ in fps]); s = np.stack([c for _, _, c in fps])
    np.savez_compressed(path, h=h, hf=hf, s=s)
    return h, hf, s

def audit(src_items, src_fp, tgt_items, tgt_fp, src_name, tgt_name):
    hs, _, ss = src_fp; ht, htf, st = tgt_fp
    rows = []
    for j in tqdm(range(len(tgt_items)), desc=f"  Comparing {tgt_name} vs {src_name}"):
        ham = np.minimum(POPCOUNT[np.bitwise_xor(hs, ht[j])].sum(1), POPCOUNT[np.bitwise_xor(hs, htf[j])].sum(1))
        i = int(ham.argmin()); best = int(ham[i])
        corr = np.nan
        if best <= HAM_CAND:
            cands = np.where(ham <= HAM_CAND)[0]
            flip = st[j].reshape(64, 64)[:, ::-1].ravel()
            cs = np.maximum(ss[cands] @ st[j], ss[cands] @ flip) / st.shape[1]
            i = int(cands[cs.argmax()]); corr = float(cs.max())
        dup = best <= HAM_CAND and corr >= CORR_THR
        rows.append({"target": tgt_name, "source": src_name, "tgt_path": tgt_items[j][0],
                     "tgt_class": IDX2CLASS[tgt_items[j][1]], "min_hamming": best, "corr": corr,
                     "match_path": src_items[i][0], "duplicate": bool(dup)})
    return pd.DataFrame(rows)

banner("1.4 Duplicate audit", "First run: ~5-15 min (reads ~15,000 images). Later runs: seconds (cached).")
overlap_csv = OUT_DIR / "overlap_audit.csv"
if overlap_csv.exists() and overlap_csv.stat().st_size > 10 and set(pd.read_csv(overlap_csv).tgt_path) == {p for p, _ in BRISC_ITEMS + MEND_ITEMS + BD_ITEMS + MSOUD_TRAIN + MSOUD_TEST}:
    df_overlap = pd.read_csv(overlap_csv)
else:
    t0 = time.time()
    FIG_ITEMS_ALL = list(zip(df3.filepath, df3.label))          # all 3,064 Figshare tumour images
    fp_fig = fingerprints(FIG_ITEMS_ALL, "figshare")
    fp_brisc = fingerprints(BRISC_ITEMS, "brisc") if BRISC_ITEMS else None
    fp_mend = fingerprints(MEND_ITEMS, "mendeley") if MEND_ITEMS else None
    parts = []
    if BRISC_ITEMS: parts.append(audit(FIG_ITEMS_ALL, fp_fig, BRISC_ITEMS, fp_brisc, "Figshare", "BRISC"))
    if MEND_ITEMS:  parts.append(audit(FIG_ITEMS_ALL, fp_fig, MEND_ITEMS, fp_mend, "Figshare", "Mendeley"))
    if BRISC_ITEMS and MEND_ITEMS:
        parts.append(audit(BRISC_ITEMS, fp_brisc, MEND_ITEMS, fp_mend, "BRISC", "Mendeley"))
    if BD_ITEMS:                                              # the new hospital must not contain known images
        fp_bd = fingerprints(BD_ITEMS, "bdneuro")
        parts.append(audit(FIG_ITEMS_ALL, fp_fig, BD_ITEMS, fp_bd, "Figshare", "BDNeuro"))
        if BRISC_ITEMS: parts.append(audit(BRISC_ITEMS, fp_brisc, BD_ITEMS, fp_bd, "BRISC", "BDNeuro"))
        if MEND_ITEMS:  parts.append(audit(MEND_ITEMS, fp_mend, BD_ITEMS, fp_bd, "Mendeley", "BDNeuro"))
    if MSOUD_TRAIN and MSOUD_TEST:                            # the most-used Kaggle brain-tumour dataset
        fp_mtr = fingerprints(MSOUD_TRAIN, "msoud_train"); fp_mte = fingerprints(MSOUD_TEST, "msoud_test")
        parts.append(audit(FIG_ITEMS_ALL, fp_fig, MSOUD_TRAIN, fp_mtr, "Figshare", "Masoud-train"))
        parts.append(audit(FIG_ITEMS_ALL, fp_fig, MSOUD_TEST, fp_mte, "Figshare", "Masoud-test"))
        parts.append(audit(MSOUD_TRAIN, fp_mtr, MSOUD_TEST, fp_mte, "Masoud-train", "Masoud-test"))  # train/test leakage
    df_overlap = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df_overlap.to_csv(overlap_csv, index=False)
    print(f"Audit done in {(time.time()-t0)/60:.1f} min")

if len(df_overlap):
    split_of = dict(zip(df3.filepath, np.where(df3.role == "test", "test", "train/val")))
    df_overlap["match_split"] = df_overlap.match_path.map(split_of).fillna("-")
    tab = (df_overlap.groupby(["target", "source", "tgt_class"]).duplicate.agg(["sum", "size"])
           .assign(pct=lambda d: (100 * d["sum"] / d["size"]).round(1)))
    print("Share of each dataset that is a near-copy of another dataset (pct = %):")
    print(tab); tab.to_csv(OUT_DIR / "tables" / "table_overlap.csv")
    print("\nFigshare split of matched images (BRISC/Mendeley duplicates):")
    print(df_overlap[df_overlap.duplicate & (df_overlap.source == "Figshare")]
          .groupby(["target", "match_split"]).size())

def clean_items(items, target):
    """Items of `target` that are NOT duplicates of Figshare (and, for Mendeley, not of BRISC)."""
    if not len(df_overlap):
        return list(items)
    bad = set(df_overlap[(df_overlap.target == target) & df_overlap.duplicate].tgt_path)
    return [(p, y) for p, y in items if p not in bad]

BRISC_CLEAN = clean_items(BRISC_ITEMS, "BRISC")
MEND_CLEAN = clean_items(MEND_ITEMS, "Mendeley")
print(f"\nBRISC: {len(BRISC_ITEMS)} -> {len(BRISC_CLEAN)} after deduplication")
print(f"Mendeley: {len(MEND_ITEMS)} -> {len(MEND_CLEAN)} after deduplication")
BD_CLEAN = clean_items(BD_ITEMS, "BDNeuro")
print(f"BDNeuro: {len(BD_ITEMS)} -> {len(BD_CLEAN)} after deduplication  "
      f"(class counts {dict(Counter(IDX2CLASS[y] for _, y in BD_CLEAN))})")
if MSOUD_TEST:
    mt = df_overlap[df_overlap.target == "Masoud-test"]
    for src in ["Figshare", "Masoud-train"]:
        d = mt[mt.source == src]
        print(f"Masoud TEST images that are near-copies of {src}: {int(d.duplicate.sum())} / {len(d)} "
              f"({100 * d.duplicate.mean():.1f}%)")
if BD_ITEMS and len(BD_CLEAN) < 0.8 * len(BD_ITEMS):
    print("  WARNING: BDNeuro contains many copies of public data -> it is evaluated on its unique images only.")

# %%
# ─── 1.5 Visual check of the audit (send these pairs to reviewers / supplement) ───
def show_pairs(df, title, fname, n=8, seed=0):
    if len(df) == 0:
        print("no pairs for", title); return
    df = df.sample(min(n, len(df)), random_state=seed)
    fig, ax = plt.subplots(2, len(df), figsize=(1.6 * len(df), 3.4))
    ax = np.array(ax).reshape(2, -1)
    for c, r in enumerate(df.itertuples()):
        ax[0, c].imshow(Image.open(r.tgt_path).convert("L"), cmap="gray")
        ax[1, c].imshow(Image.open(r.match_path).convert("L"), cmap="gray")
        ax[0, c].set_title(f"{r.target}\nH={r.min_hamming} r={r.corr:.2f}", fontsize=6)
        ax[1, c].set_title(r.source, fontsize=6)
        for a in ax[:, c]: a.axis("off")
    fig.suptitle(title, fontsize=9); fig.tight_layout()
    fig.savefig(FIG_DIR / fname); plt.show()

def pair_sheet(df, title, fname, n=16, seed=0):
    """n random flagged pairs in 2 rows of n/2: each pair = dataset image (top) + its Figshare match (below)."""
    if len(df) == 0:
        print("  no pairs for", title); return
    df = df.sample(min(n, len(df)), random_state=seed); cols = min(8, len(df)); rows = 2 * int(np.ceil(len(df) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.7 * rows), squeeze=False)
    for a in ax.ravel(): a.axis("off")
    for k, r in enumerate(df.itertuples()):
        rr, c = 2 * (k // cols), k % cols
        ax[rr, c].imshow(Image.open(r.tgt_path).convert("L"), cmap="gray")
        ax[rr + 1, c].imshow(Image.open(r.match_path).convert("L"), cmap="gray")
        ax[rr, c].set_title(f"{r.target} ({r.tgt_class[:4]})", fontsize=6)
        ax[rr + 1, c].set_title(f"{r.source}  r={r.corr:.3f}", fontsize=6)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(FIG_DIR / fname, bbox_inches="tight"); plt.show()

if len(df_overlap):
    for tgt in df_overlap.target.unique():
        src = "Figshare" if tgt != "Masoud-test" else "Masoud-train"
        d = df_overlap[(df_overlap.target == tgt) & (df_overlap.source == src) & df_overlap.duplicate]
        pair_sheet(d, f"{tgt}: 16 random images flagged as near-copies of {src} (pairs: top = {tgt}, below = match)",
                   f"fig_pairs_{tgt.lower().replace('-', '_')}.png")
    border = df_overlap[(~df_overlap.duplicate) & (df_overlap["corr"].between(0.85, CORR_THR))]
    pair_sheet(border, "Borderline pairs (0.85 <= r < 0.95) - NOT counted as copies (check visually)", "fig_pairs_borderline.png")

# %%
# ─── 1.6 Tumour masks for every Figshare slice (from the ORIGINAL .mat files) ───
# The PNG dataset has no masks. Each PNG is matched to its .mat file by pixel content
# (correlation on 64x64 thumbnails, within the same tumour class), the orientation is detected
# automatically, and the mask is saved at 224x224. Cached: later runs take 1 second.
MASKS = {}                                          # PNG path -> bool (224, 224)
mask_cache = OUT_DIR / "figshare_masks_224.npz"
MAT_LABEL = {1: "meningioma", 2: "glioma", 3: "pituitary"}

def png224(path):                                   # same pixels as load_u8 (defined in 2.4)
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.uint8)

def read_mat(path):
    try:
        import h5py
        with h5py.File(path, "r") as h:
            cj = h["cjdata"]
            lab = int(np.array(cj["label"]).squeeze())
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel())
            return lab, pid, np.array(cj["image"]).T.astype(np.float32), np.array(cj["tumorMask"]).T > 0
    except OSError:                                 # old (non-HDF5) .mat version
        from scipy.io import loadmat
        cj = loadmat(path)["cjdata"][0, 0]
        return (int(cj["label"].squeeze()), str(cj["PID"].squeeze()),
                cj["image"].astype(np.float32), cj["tumorMask"] > 0)

def thumb(a, s=64):
    t = np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((s, s), Image.BILINEAR), np.float64).ravel()
    return (t - t.mean()) / (t.std() + 1e-8)

DIHEDRAL = [lambda a: a, np.fliplr, np.flipud, lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2),
            lambda a: np.rot90(a, 3), np.transpose, lambda a: np.rot90(a.T, 2)]

if not mask_cache.exists() and MASK_NPZ_INPUT is not None:
    shutil.copy2(MASK_NPZ_INPUT, mask_cache)              # uploaded mask file: skips ~25 min of .mat reading
if mask_cache.exists():
    z = np.load(mask_cache, allow_pickle=False)
    bits = np.unpackbits(z["bits"], axis=1)[:, :IMG_SIZE * IMG_SIZE].reshape(-1, IMG_SIZE, IMG_SIZE).astype(bool)
    stems = list(z["stems"])
    by_stem = {Path(p).stem: p for p in df3.filepath}
    MASKS = {by_stem[s]: b for s, b in zip(stems, bits) if s in by_stem}
    print(f"Tumour masks loaded from cache: {len(MASKS)} of {len(df3)} Figshare slices")
elif any(FIGSHARE_MAT_DIR.glob("brainTumorDataPublic_*")):
    banner("1.6 Matching PNG slices to the original .mat files (tumour masks)", "first run only: ~3-6 min")
    mats = sorted(FIGSHARE_MAT_DIR.glob("brainTumorDataPublic_*/*.mat"))
    mat_rows = []
    for f in tqdm(mats, desc="  Reading .mat files"):
        lab, pid, img, mk = read_mat(f)
        mat_rows.append((f.name, MAT_LABEL.get(lab), pid, img, mk))
    # 1) orientation: which of 8 flips/rotations makes the .mat images look like the PNGs?
    sample = df3.sample(min(60, len(df3)), random_state=0)
    png_t = {p: thumb(png224(p)) for p in sample.filepath}
    scores = []
    for op in DIHEDRAL:
        mt = np.stack([thumb(op(r[3])) for r in mat_rows[::5]])
        scores.append(np.mean([np.max(mt @ png_t[p]) / png_t[p].size for p in png_t]))
    best_op = int(np.argmax(scores)); op = DIHEDRAL[best_op]
    print(f"  orientation: transform #{best_op} (mean best correlation {scores[best_op]:.3f})")
    # 2) match every PNG to the best .mat of the same class
    n_ok, corrs, pid_agree = 0, [], []
    for cls in CLASSES:
        rows_c = [r for r in mat_rows if r[1] == cls]
        if not rows_c:
            continue
        M = np.stack([thumb(op(r[3])) for r in rows_c])
        sub = df3[df3.label4 == cls]
        Pt = np.stack([thumb(png224(p)) for p in tqdm(sub.filepath, desc=f"  Matching {cls}", leave=False)])
        C = Pt @ M.T / Pt.shape[1]
        best = C.argmax(1)
        for (p, pid), j, c in zip(zip(sub.filepath, sub.pid), best, C[np.arange(len(best)), best]):
            corrs.append(c)
            if c >= 0.95:
                mk = np.asarray(Image.fromarray(op(rows_c[j][4]).astype(np.uint8) * 255)
                                .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
                MASKS[p] = mk; n_ok += 1
                pid_agree.append(str(rows_c[j][2]).strip() == str(pid))
    corrs = np.array(corrs)
    print(f"  matched {n_ok} of {len(df3)} slices (correlation >= 0.95); median r = {np.median(corrs):.4f}; "
          f"patient ID agrees in {100 * np.mean(pid_agree):.1f}% of matches")
    stems = np.array([Path(p).stem for p in MASKS])
    bits = np.packbits(np.stack(list(MASKS.values())).reshape(len(MASKS), -1), axis=1)
    np.savez_compressed(mask_cache, stems=stems, bits=bits)
else:
    print("Figshare .mat files not found -> CTR-Net cannot be trained (add the 'figshare-original-mat' input).")

if MASKS:
    cover = pd.Series({c: np.mean([p in MASKS for p in df3[df3.label4 == c].filepath]) for c in CLASSES})
    print("  mask coverage per class (%):", (100 * cover).round(1).to_dict())
    area = pd.Series({c: np.mean([MASKS[p].mean() for p in df3[df3.label4 == c].filepath if p in MASKS]) for c in CLASSES})
    print("  mean tumour area (% of image):", (100 * area).round(2).to_dict())
    rng_m = np.random.default_rng(1)
    show = [p for c in CLASSES for p in rng_m.choice([q for q in df3[df3.label4 == c].filepath if q in MASKS], 2, replace=False)]
    fig, ax = plt.subplots(1, len(show), figsize=(1.7 * len(show), 2))
    for a, p in zip(ax, show):
        a.imshow(png224(p), cmap="gray"); a.contour(MASKS[p], levels=[0.5], colors="lime", linewidths=0.8)
        a.set_title(Path(p).parent.name, fontsize=7); a.axis("off")
    fig.suptitle("Check: expert tumour outline from the .mat file on the PNG slice (must sit on the tumour)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_mask_check.png"); plt.show()

# %%
# ─── 1.7 Figure: how much of the "external" data is really Figshare? ───
if len(df_overlap):
    leak = (df_overlap.groupby(["target", "source", "tgt_class"]).duplicate.mean() * 100).reset_index()
    pairs_ = [(t, "Figshare") for t in ["BRISC", "Mendeley", "BDNeuro"] if (leak.target == t).any()]
    fig, ax = plt.subplots(1, len(pairs_), figsize=(4.5 * len(pairs_), 3.2), squeeze=False); ax = ax[0]
    for a, (tgt, src) in zip(ax, pairs_):
        d = leak[(leak.target == tgt) & (leak.source == src)].set_index("tgt_class").reindex(CLASSES)
        bars = a.bar(CLASSES, d.duplicate, color=[CLASS_COLORS[c] for c in CLASSES])
        for b, v in zip(bars, d.duplicate):
            a.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
        a.set(ylim=(0, 110), ylabel="% of images that are near-duplicates", title=f"{tgt} images found in {src}")
    n_all = {"BRISC": len(BRISC_ITEMS), "Mendeley": len(MEND_ITEMS), "BDNeuro": len(BD_ITEMS)}
    n_cln = {"BRISC": len(BRISC_CLEAN), "Mendeley": len(MEND_CLEAN), "BDNeuro": len(BD_CLEAN)}
    n_all = {k: v for k, v in n_all.items() if v}
    fig.suptitle("Cross-dataset duplication. Unique images left: " +
                 ", ".join(f"{k} {n_cln[k]}/{n_all[k]}" for k in n_all), fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_duplicate_leakage.png"); plt.show()

# %%
# ─── 1.8 Tumour-hiding versions of every slice (used for CTR training AND for the tumour-reliance test) ───
# region  = expert mask + small margin
# inpaint = tumour region filled from the surrounding tissue          (used in CTR TRAINING)
# swap    = tumour region replaced by a same-shape patch of normal brain from the same slice (TEST ONLY:
#           a different hiding method than in training, so CTR cannot pass the test by learning our trick)
# random  = a same-shape region elsewhere in the brain filled (control: should change nothing)
# skull   = brain interior removed, only skull outline + slice position left
# tonly   = only the tumour region kept (supplementary)
import cv2

def brain_mask(u8):
    _, th = cv2.threshold(cv2.GaussianBlur(u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return th > 0
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m); cv2.drawContours(filled, cnts, -1, 1, -1)
    return filled > 0

HIDE_KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def shifted_region(region, bm, rng, tries=200):
    """Offset (dy, dx) that moves `region` to a place inside the brain that does not touch the tumour."""
    ys, xs = np.where(region)
    if len(ys) == 0:
        return None
    near = cv2.dilate(region.astype(np.uint8), HIDE_KER, iterations=2).astype(bool)
    for _ in range(tries):
        dy, dx = (int(v) for v in rng.integers(-IMG_SIZE // 2, IMG_SIZE // 2, 2))
        y2, x2 = ys + dy, xs + dx
        if y2.min() < 0 or x2.min() < 0 or y2.max() >= IMG_SIZE or x2.max() >= IMG_SIZE:
            continue
        r = np.zeros_like(region); r[y2, x2] = True
        if not (r & near).any() and bm[r].mean() >= 0.95:
            return dy, dx, r
    return None

def hide_versions(img, mk, rng):
    reg = cv2.dilate(mk.astype(np.uint8), HIDE_KER).astype(bool)
    bm = brain_mask(img)
    inp = cv2.inpaint(img, reg.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    sh = shifted_region(reg, bm, rng)
    if sh is not None:
        dy, dx, rr = sh
        ys, xs = np.where(reg)
        swap = img.copy(); swap[ys, xs] = img[ys + dy, xs + dx]
        rnd = cv2.inpaint(img, rr.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    else:                                     # tumour too large to move: control not available for this slice
        swap, rnd = inp, img
    interior = cv2.erode(bm.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    skull = img.copy(); skull[interior | reg] = 0
    return {"region": reg, "inpaint": inp, "swap": swap, "random": rnd, "skull": skull,
            "tonly": np.where(reg, img, 0).astype(np.uint8), "has_control": sh is not None}

HIDE = {}
if MASKS:
    banner("1.8 Building tumour-hidden versions of every slice", f"{len(MASKS)} slices (~1-2 min)")
    rng_h = np.random.default_rng(SPLIT_SEED)
    for p in tqdm(sorted(MASKS), desc="  building", mininterval=5):
        HIDE[p] = hide_versions(png224(p), MASKS[p], rng_h)
    print(f"  done: {len(HIDE)} slices | random-spot control possible for "
          f"{100 * np.mean([v['has_control'] for v in HIDE.values()]):.1f}%")
    rng_s = np.random.default_rng(3)
    show = [rng_s.choice([p for p in df3[df3.label4 == c].filepath if p in HIDE and HIDE[p]["has_control"]]) for c in CLASSES]
    cols = [("Original", None), ("Tumour hidden\n(inpaint, training)", "inpaint"), ("Tumour hidden\n(swap, test)", "swap"),
            ("Random spot hidden\n(control)", "random"), ("Skull / position\nonly", "skull"), ("Tumour only", "tonly")]
    fig, ax = plt.subplots(3, len(cols), figsize=(2.1 * len(cols), 6.6))
    for r, p in enumerate(show):
        for c, (title, key) in enumerate(cols):
            ax[r, c].imshow(png224(p) if key is None else HIDE[p][key], cmap="gray", vmin=0, vmax=255)
            if key is None:
                ax[r, c].contour(MASKS[p], levels=[0.5], colors="lime", linewidths=0.7)
            ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(title, fontsize=8)
        ax[r, 0].text(-10, IMG_SIZE / 2, CLASSES[r], rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Tumour-hiding test: versions of the same slice (green = expert tumour outline)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_hiding_versions.png", bbox_inches="tight"); plt.show()

# %%
# ─── 1.9 Slice plane (axial / coronal / sagittal) - semi-automatic labelling ───
# Figshare slices come in 3 viewing planes. A pretrained image model groups the slices into 12 visual clusters;
# the plane of each cluster was read from the Run 4 sheet (PLANE_BY_SIZE in cell 0.2).
# "mixed" clusters are left out of the plane analysis.
from sklearn.cluster import KMeans
PLANE_OF = {}
plane_csv = OUT_DIR / "plane_clusters.csv"
if plane_csv.exists() and len(pd.read_csv(plane_csv)) == len(df3):
    df_plane = pd.read_csv(plane_csv)
else:
    banner("1.9 Grouping slices by viewing plane", "pretrained image features + 12 clusters (~2 min)")
    fe = timm.create_model("convnext_tiny", pretrained=True, num_classes=0).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        arrs_all = [png224(p) for p in df3.filepath]
        for i in range(0, len(arrs_all), 64):
            x = torch.from_numpy(np.stack(arrs_all[i:i + 64])).to(DEVICE).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
            m_ = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1); s_ = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
            feats.append(fe((x - m_) / s_).float().cpu().numpy())
    del fe
    feats = np.concatenate(feats); feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    df_plane = pd.DataFrame({"filepath": df3.filepath.values, "cluster": KMeans(12, n_init=10, random_state=0).fit_predict(feats)})
    df_plane.to_csv(plane_csv, index=False)
rng_p = np.random.default_rng(0)
fig, ax = plt.subplots(12, 10, figsize=(10, 12.6))
for c in range(12):
    paths = df_plane[df_plane.cluster == c].filepath.values
    pick = rng_p.choice(paths, min(10, len(paths)), replace=False) if len(paths) else []
    for j in range(10):
        ax[c, j].axis("off")
        if j < len(pick):
            ax[c, j].imshow(png224(pick[j]), cmap="gray")
    ax[c, 0].text(-20, 112, f"cluster {c}\n(n={len(paths)})\n{PLANE_BY_SIZE.get(len(paths), '?')}", ha="right", va="center", fontsize=7)
fig.suptitle("Viewing-plane clusters (each row = one cluster; its plane is printed below the sheet)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "plane_cluster_sheet.png", bbox_inches="tight"); plt.show()
sizes = df_plane.cluster.value_counts().to_dict()                     # cluster -> number of slices
PLANE_MAP = {c: PLANE_BY_SIZE[n] for c, n in sizes.items() if n in PLANE_BY_SIZE}
if len(PLANE_MAP) == 12:
    df_plane["plane"] = df_plane.cluster.map(PLANE_MAP)
    PLANE_OF = dict(zip(df_plane.filepath, df_plane.plane))
    print("✔ Clusters recognised (same sizes as in Run 4). Slices per plane:", df_plane.plane.value_counts().to_dict())
else:
    PLANE_MAP = {}
    print("⚠ The clusters differ from Run 4 (sizes " + str(sorted(sizes.values())) + "). The plane analysis (6.9) is "
          "skipped. Send me this sheet and I will update PLANE_BY_SIZE.")
