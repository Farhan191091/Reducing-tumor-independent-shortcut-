"""
BraTS 2023 -> TUMOUR-FREE 2D slices (same patients as brats_2d_slices)
=====================================================================
Run this on YOUR PC. For the same 200 GLI + 200 MEN patients used before, it saves
up to 3 axial slices that contain NO tumour at all (far away from the tumour, but
with plenty of brain visible). A trustworthy classifier should be UNSURE on them.

HOW TO USE
  1. pip install nibabel numpy pillow        (already done last time)
  2. Copy the SAME folder paths you used in brats_extract.py into SETTINGS below.
  3. python brats_free_extract.py
  4. Upload brats_free_slices.zip to Kaggle as a new dataset: brats-free-slices
"""

import csv
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    from PIL import Image
except ImportError:
    sys.exit("Missing packages. Run:  pip install nibabel numpy pillow")

# ───────────────────────────── SETTINGS ─────────────────────────────
# Use EXACTLY the same paths as in brats_extract.py
GLI_ROOTS = [r"D:\BraTS\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"]
MEN_ROOTS = [
    r"D:\BraTS\BraTS-MEN-Train",
    r"D:\BraTS\MEN-TRAIN-FIX-V4",          # fix folder LAST
]
OUT_DIR = r"D:\BraTS\brats_free_slices"

N_PATIENTS = 200          # same as before -> same patients (same seed)
SEED = 42
ROTATE_K = 3              # same as the final brats_extract.py (nose up)
SLICES_PER_PATIENT = 3
MIN_DIST_TO_TUMOUR = 10   # slices this far (in slices) from any tumour voxel
MIN_BRAIN_FRACTION = 0.6  # slice must show >= 60% of the largest brain cross-section
MIN_GAP = 5               # chosen slices at least this many slices apart
# ─────────────────────────────────────────────────────────────────────

CASE_RE = re.compile(r"BraTS-(GLI|MEN)-(\d{5})-(\d{3})$")


def find_cases(roots):
    cases = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"  WARNING: folder not found, skipped: {root}")
            continue
        for t1c in root.rglob("*-t1c.nii*"):
            case_id = t1c.name.split("-t1c")[0]
            if CASE_RE.match(case_id):
                cases[case_id] = t1c.parent
    return cases


def one_case_per_patient(cases):
    by_patient = {}
    for case_id in sorted(cases):
        _, pid, tp = CASE_RE.match(case_id).groups()
        if pid not in by_patient or tp == "000":
            by_patient[pid] = case_id
    return by_patient


def file_for(folder, case_id, suffix):
    hits = sorted(Path(folder).glob(f"{case_id}-{suffix}.nii*"))
    return hits[0] if hits else None


def to_uint8(slice_2d, lo, hi):
    x = np.clip((slice_2d - lo) / max(hi - lo, 1e-6), 0, 1)
    return (x * 255).astype(np.uint8)


def pick_free_slices(brain_area, tumour_z):
    """Tumour-free slices: far from the tumour, large brain area, spread apart."""
    nz = len(brain_area)
    if len(tumour_z) == 0:
        return []
    dist = np.array([np.min(np.abs(tumour_z - z)) for z in range(nz)])
    ok = (dist >= MIN_DIST_TO_TUMOUR) & (brain_area >= MIN_BRAIN_FRACTION * brain_area.max())
    chosen = []
    for z in np.argsort(-brain_area):           # largest brain cross-sections first
        if ok[z] and all(abs(int(z) - c) >= MIN_GAP for c in chosen):
            chosen.append(int(z))
        if len(chosen) == SLICES_PER_PATIENT:
            break
    return sorted(chosen), dist


def process(name, label, roots, out_dir, writer):
    print(f"\n=== {name.upper()} ({label} patients, tumour-free slices) ===")
    cases = find_cases(roots)
    patients = one_case_per_patient(cases)
    print(f"  cases found: {len(cases)} | unique patients: {len(patients)}")
    if not patients:
        print("  Nothing to do - check the folder paths in SETTINGS.")
        return
    pids = sorted(patients)
    random.Random(SEED).shuffle(pids)
    pids = pids[:N_PATIENTS]
    (out_dir / name / "images").mkdir(parents=True, exist_ok=True)

    saved, n_pat, previews = 0, 0, []
    for i, pid in enumerate(pids, 1):
        case_id = patients[pid]; folder = cases[case_id]
        t1c_f, seg_f = file_for(folder, case_id, "t1c"), file_for(folder, case_id, "seg")
        if t1c_f is None or seg_f is None:
            continue
        img = np.asarray(nib.load(str(t1c_f)).dataobj, dtype=np.float32)
        seg = np.asarray(nib.load(str(seg_f)).dataobj).astype(np.uint8)
        if img.shape != seg.shape:
            continue
        brain = img[img > 0]
        lo, hi = (np.percentile(brain, [0.5, 99.5]) if brain.size else (0, 1))
        tumour_z = np.where((seg > 0).sum((0, 1)) > 0)[0]
        brain_area = (img > 0).sum((0, 1)).astype(float)
        res = pick_free_slices(brain_area, tumour_z)
        if not res or not res[0]:
            continue
        zs, dist = res
        n_pat += 1
        for s, z in enumerate(zs, 1):
            fname = f"{name}_{pid}_free{s}.png"
            im = to_uint8(np.rot90(img[:, :, z], ROTATE_K), lo, hi)
            Image.fromarray(im).save(out_dir / name / "images" / fname)
            writer.writerow({"file": f"{name}/images/{fname}", "dataset": name,
                             "patient_tumour_type": label, "patient_id": f"{name}_{pid}",
                             "case_id": case_id, "slice_z": z,
                             "slices_from_tumour": int(dist[z]),
                             "brain_fraction": round(float(brain_area[z] / brain_area.max()), 3),
                             "height": im.shape[0], "width": im.shape[1]})
            saved += 1
            if s == 1 and len(previews) < 12:
                previews.append(im)
        if i % 25 == 0:
            print(f"  {i}/{len(pids)} patients done ({saved} slices)")
    if len(previews) >= 4:
        grid = np.concatenate([np.concatenate(previews[r:r + 4], axis=1)
                               for r in range(0, len(previews) - 3, 4)], axis=0)
        Image.fromarray(grid).save(out_dir / f"preview_free_{name}.png")
    print(f"  saved {saved} tumour-free slices from {n_pat} patients")


def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["file", "dataset", "patient_tumour_type", "patient_id", "case_id", "slice_z",
              "slices_from_tumour", "brain_fraction", "height", "width"]
    with open(out_dir / "manifest_free.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        process("gli", "glioma", GLI_ROOTS, out_dir, writer)
        process("men", "meningioma", MEN_ROOTS, out_dir, writer)
    zip_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"\nDONE. Open preview_free_gli.png / preview_free_men.png: brains must be upright and show "
          f"NO tumour.\nUpload this file to Kaggle as dataset 'brats-free-slices': {zip_path}")


if __name__ == "__main__":
    main()
