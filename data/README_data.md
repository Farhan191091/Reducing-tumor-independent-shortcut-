# Data sources (not redistributed)

| Dataset | Use | Source |
|---|---|---|
| Figshare brain tumor dataset (Cheng, 2017) | Development and internal test | https://doi.org/10.6084/m9.figshare.1512427 |
| BraTS 2023 adult glioma and meningioma (training sets) | External test, tumor free slices | BraTS 2023 challenge on the Synapse platform (free registration required) |
| BDNeuro-MRI, version 8 (Hira et al., 2026) | External test | https://doi.org/10.17632/zwr4ntf94j.8 |
| Mendeley brain tumor MRI, version 1 (Hira et al., 2025) | External test | https://doi.org/10.17632/zwr4ntf94j.1 |
| BRISC (Fateh et al., 2026) | Duplicate audit, explanation analysis | https://doi.org/10.1038/s41597-026-06753-y |
| Kaggle brain tumor MRI, version 2 (Nickparvar, 2021) | Duplicate audit only | https://doi.org/10.34740/KAGGLE/DSV/2645886 |

Files in this folder:

* `split_3class.csv`: patient level split (test set and five folds) of the 3,064 Figshare slices.
* `overlap_audit.csv`: every external image with its nearest match in each source collection, Hamming distance, correlation and duplicate decision.
* `evaluated_images_*.csv`: external images without a near duplicate, i.e. the images evaluated in the paper.
* `manifest_brats.csv`, `manifest_free.csv`: BraTS tumor slices (filter `tc_pixels >= 100`) and tumor free slices, as produced by `data_prep/`.
