"""
BraTS 2023 -> small 2D PNG dataset for MedTriMambaKAN external testing
=====================================================================
Run this on YOUR PC (not Kaggle). It reads only the post-contrast T1
(t1c) and the tumour mask (seg) of each patient, picks the 3 axial
slices with the largest tumour core, and saves them as small PNGs.

HOW TO USE (step by step)
  1. Install once:      pip install nibabel numpy pillow
  2. Edit the 4 paths in "SETTINGS" below.
  3. Run:               python brats_extract.py
  4. Upload the ZIP it creates (brats_2d_slices.zip) to Kaggle as a
     new dataset named  brats-2d-slices.

Output layout (inside OUT_DIR):
  gli/images/gli_00005_s1.png      glioma slice (8-bit grayscale)
  gli/masks_tc/gli_00005_s1.png    tumour-core mask (labels 1+3)
  gli/masks_wt/gli_00005_s1.png    whole-tumour mask (labels 1+2+3)
  men/...                          same for meningioma
  manifest.csv                     one row per saved slice
  preview_gli.png / preview_men.png   quick visual check
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
# Folder(s) where you UNZIPPED the GLI training zip.
GLI_ROOTS = [r"D:\BraTS\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"]

# Folder(s) where you UNZIPPED the MEN training zip. Put the FIX folder
# LAST: if a case exists in both, the later folder (the fix) is used.
MEN_ROOTS = [
    r"D:\BraTS\ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData",
    r"D:\BraTS\BraTS-MEN-TRAIN-FIX-V4",
]

OUT_DIR = r"D:\BraTS\brats_2d_slices"   # where PNGs are written

N_PATIENTS = 200        # patients per dataset (random, fixed seed)
SLICES_PER_PATIENT = 3  # axial slices with the largest tumour core
MIN_GAP = 4             # chosen slices must be >= this many slices apart
SEED = 42
ROTATE_K = 3            # np.rot90 turns used for the paper (nose up; same as brats_free_extract.py)
# ─────────────────────────────────────────────────────────────────────

CASE_RE = re.compile(r"BraTS-(GLI|MEN)-(\d{5})-(\d{3})$")


def find_cases(roots):
    """Return {case_id: folder}. Later roots override earlier ones."""
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
    """Keep timepoint -000 when available (avoid same patient twice)."""
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


def pick_slices(tc_area, wt_area, k, min_gap):
    score = np.where(tc_area > 0, tc_area, 0).astype(float)
    if score.max() == 0:              # no tumour core: fall back to whole tumour
        score = wt_area.astype(float)
    chosen = []
    for z in np.argsort(-score):
        if score[z] <= 0 or len(chosen) == k:
            break
        if all(abs(int(z) - c) >= min_gap for c in chosen):
            chosen.append(int(z))
    return sorted(chosen)


def process(name, label, roots, out_dir, writer):
    print(f"\n=== {name.upper()} ({label}) ===")
    cases = find_cases(roots)
    patients = one_case_per_patient(cases)
    print(f"  cases found: {len(cases)} | unique patients: {len(patients)}")
    if not patients:
        print("  Nothing to do - check the folder paths in SETTINGS.")
        return []

    pids = sorted(patients)
    random.Random(SEED).shuffle(pids)
    pids = pids[:N_PATIENTS]

    for sub in ("images", "masks_tc", "masks_wt"):
        (out_dir / name / sub).mkdir(parents=True, exist_ok=True)

    saved, previews = 0, []
    for i, pid in enumerate(pids, 1):
        case_id = patients[pid]
        folder = cases[case_id]
        t1c_f = file_for(folder, case_id, "t1c")
        seg_f = file_for(folder, case_id, "seg")
        if t1c_f is None or seg_f is None:
            print(f"  skip {case_id}: t1c or seg missing")
            continue
        img = np.asarray(nib.load(str(t1c_f)).dataobj, dtype=np.float32)
        seg = np.asarray(nib.load(str(seg_f)).dataobj).astype(np.uint8)
        if img.shape != seg.shape:
            print(f"  skip {case_id}: shape mismatch {img.shape} vs {seg.shape}")
            continue

        brain = img[img > 0]
        lo, hi = (np.percentile(brain, [0.5, 99.5]) if brain.size else (0, 1))
        tc = np.isin(seg, (1, 3))                 # tumour core
        wt = seg > 0                              # whole tumour
        zs = pick_slices(tc.sum((0, 1)), wt.sum((0, 1)),
                         SLICES_PER_PATIENT, MIN_GAP)

        for s, z in enumerate(zs, 1):
            fname = f"{name}_{pid}_s{s}.png"
            im = to_uint8(np.rot90(img[:, :, z], ROTATE_K), lo, hi)
            m_tc = (np.rot90(tc[:, :, z], ROTATE_K) * 255).astype(np.uint8)
            m_wt = (np.rot90(wt[:, :, z], ROTATE_K) * 255).astype(np.uint8)
            Image.fromarray(im).save(out_dir / name / "images" / fname)
            Image.fromarray(m_tc).save(out_dir / name / "masks_tc" / fname)
            Image.fromarray(m_wt).save(out_dir / name / "masks_wt" / fname)
            writer.writerow({
                "file": f"{name}/images/{fname}", "dataset": name,
                "label": label, "patient_id": f"{name}_{pid}",
                "case_id": case_id, "slice_z": z, "slice_rank": s,
                "tc_pixels": int(m_tc.astype(bool).sum()),
                "wt_pixels": int(m_wt.astype(bool).sum()),
                "height": im.shape[0], "width": im.shape[1],
            })
            saved += 1
            if s == 1 and len(previews) < 12:
                previews.append(im)
        if i % 25 == 0:
            print(f"  {i}/{len(pids)} patients done ({saved} slices)")

    if previews:
        grid = np.concatenate([np.concatenate(previews[r:r + 4], axis=1)
                               for r in range(0, len(previews) - 3, 4)], axis=0)
        Image.fromarray(grid).save(out_dir / f"preview_{name}.png")
    print(f"  saved {saved} slices from {len(pids)} patients")
    return pids


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["file", "dataset", "label", "patient_id", "case_id", "slice_z",
              "slice_rank", "tc_pixels", "wt_pixels", "height", "width"]
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        process("gli", "glioma", GLI_ROOTS, out_dir, writer)
        process("men", "meningioma", MEN_ROOTS, out_dir, writer)

    zip_path = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"\nDONE. Open preview_gli.png / preview_men.png to check the images "
          f"look upright (nose at the top).\nIf they look sideways or upside "
          f"down, change ROTATE_K (1, 2 or 3) and run again.\n"
          f"Upload this file to Kaggle: {zip_path}")


if __name__ == "__main__":
    main()
