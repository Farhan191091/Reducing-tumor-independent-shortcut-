**Table 3.** Internal accuracy and tumor reliance. Cross validation (CV) macro F1 is the mean ± standard deviation over all runs; test results use the five fold ensemble of seed 0 on the 479 test slices (35 patients). Tumor reliance $\rho$ (Eq. 3) uses the swap operator on all 3,064 slices with the first seed; abstained: share of hidden slices on which the model abstained; patients: patients correctly classified by majority vote; $\rho_{\mathrm{ans}}$: reliance on the answered slices only; head only: balanced accuracy when only the head outline remains (chance 33.3%). BA, balanced accuracy; CI, patient bootstrap confidence interval.

| Model | CV macro F1 (%) | Test macro F1 (%) | Patients | $\rho$ (95% CI) | Abstained (%) | $\rho_{\mathrm{ans}}$ | Head only BA (%) |
|:------------|:----------|:-------|:---------|:----------------------|:----------|:-----|:-------|
| *Standard* |  |  |  |  |  |  |  |
| Baseline | 97.1 ± 2.2 | 94.1 | 34/35 | 0.35 (0.32–0.38) | 0.0 | 0.35 | 38.9 |
| ConvNeXt-T (standard head) | 97.4 ± 2.1 | 93.7 | 34/35 | 0.29 (0.26–0.33)^\*\*\*^ | 0.0 | 0.29 | 45.9 |
| ResNet-50 | 93.8 ± 2.2^\*\*^ | 89.0 | 31/35 | 0.16 (0.12–0.19)^\*\*\*^ | 0.0 | 0.16 | 57.9 |
| EfficientNet-B0 | 94.3 ± 2.7 | 86.3^\*\*\*^ | 29/35 | 0.13 (0.10–0.16)^\*\*\*^ | 0.0 | 0.13 | 51.4 |
| DeiT-S | 96.0 ± 3.1 | 89.9 | 32/35 | 0.24 (0.20–0.27)^\*\*\*^ | 0.0 | 0.24 | 50.6 |
| *Guidance* |  |  |  |  |  |  |  |
| MTL-Seg | 97.2 ± 2.4 | 96.3 | 35/35 | 0.24 (0.20–0.27)^\*\*\*^ | 0.0 | 0.24 | 45.5 |
| RRR | 97.5 ± 2.2 | 96.1 | 35/35 | 0.30 (0.27–0.34)^\*^ | 0.0 | 0.30 | 35.5 |
| CAM-Guide | 97.4 ± 2.3 | 94.0 | 35/35 | 0.29 (0.26–0.32)^\*\*\*^ | 0.0 | 0.29 | 39.0 |
| CTR-Net | 97.4 ± 2.4 | 94.3 | 34/35 | 0.27 (0.23–0.30)^\*\*\*^ | 0.0 | 0.27 | 42.0 |
| *Input restriction* |  |  |  |  |  |  |  |
| ROI-0.2 | 93.8 ± 1.3 | 95.0 | 35/35 | 0.64 (0.60–0.68)^\*\*\*^ | 26.3 | 0.52 | 33.7^b^ |
| ROI-0.6 | 94.6 ± 1.5 | 96.3 | 35/35 | 0.55 (0.52–0.59)^\*\*\*^ | 26.3 | 0.40 | 33.6^b^ |
| ROI-1.5 | 94.5 ± 1.8 | 97.4^\*\*\*^ | 35/35 | 0.49 (0.45–0.53)^\*\*\*^ | 26.3 | 0.33 | 33.7^b^ |
| ROI-Mask | 87.3 ± 5.3 | 91.2 | 34/35 | 0.80 (0.77–0.84)^\*\*\*^ | 26.3 | 0.77 | 33.5^b^ |
| ROI-Loc | 94.1 ± 1.8 | 94.1 | 35/35 | 0.69 (0.66–0.73)^\*\*\*^ | 26.3 | 0.59 | 33.6^b^ |
| *Tumor transplantation* |  |  |  |  |  |  |  |
| TT | 94.8 ± 1.2 | 93.0 | 35/35 | 0.58 (0.55–0.62)^\*\*\*^ | 0.0 | 0.58 | 36.2 |
| *Both principles* |  |  |  |  |  |  |  |
| **ROI-0.6+TT** | **92.1 ± 1.4^\*\*^** | **91.6** | **34/35** | **0.68 (0.64–0.72)^\*\*\*^** | **26.3** | **0.59** | **33.5^b^** |
| *Skull control* |  |  |  |  |  |  |  |
| Baseline-Strip | 96.6 ± 2.2 | 93.6 | 34/35 | 0.30 (0.27–0.34)^\*^ | 0.0 | 0.30 | 31.5 |
| *Foundation models^a^* |  |  |  |  |  |  |  |
| DINOv2-S/14 | – | – | – | 0.16 | 0.0 | 0.16 | 36.0 |
| DINOv2-B/14 | – | – | – | 0.16 | 0.0 | 0.16 | 37.8 |
| BiomedCLIP | – | – | – | 0.27 | 0.0 | 0.27 | 33.8 |

Differences from the baseline after Holm correction: \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001 (CV: corrected resampled t test; test and $\rho$: patient bootstrap). ^a^ Frozen encoders with a logistic regression classifier, evaluated without cross validation (internal test balanced accuracy 83.2% to 84.8%); macro F1 and patient votes were not computed. ^b^ The model abstained on 98.6% of head only slices, so this value reflects abstention.

**Table 4.** Transfer and safety. BraTS: 1,073 slices of 368 patients; AUC for glioma versus meningioma; Recall G and Recall M, recall of glioma and meningioma. BDNeuro (227 slices) and Mendeley (1,108 slices): balanced accuracy on the images that are not near duplicates of the development data. Tumor free: share of the 911 BraTS tumor free slices answered with a calibrated maximum probability of at least 0.85. Coverage: empirical coverage of cross conformal prediction sets on BraTS at a target of 95%. All values except AUC are percentages. BA, balanced accuracy.

| Model | BraTS AUC | Recall G | Recall M | BDNeuro | Mendeley | Tumor free | Coverage |
|:-------------------|:---------|:---------|:---------|:----------|:----------|:----------|:----------|
| *Standard* |  |  |  |  |  |  |  |
| Baseline | 0.924 | 80.8 | 88.0 | 92.9 | 92.2 | 56.2 | 80.3 |
| ConvNeXt-T (standard head) | 0.912 | 82.7 | 86.4 | 93.4 | 92.8 | 55.7 | 84.1 |
| ResNet-50 | 0.845^\*\*\*^ | 70.8 | 86.2 | 91.2 | 87.4^\*\*\*^ | 37.0^\*\*\*^ | 82.6 |
| EfficientNet-B0 | 0.786^\*\*\*^ | 67.6 | 79.8 | 90.2 | 86.9^\*\*\*^ | 1.6^\*\*\*^ | 78.9 |
| DeiT-S | 0.871^\*\*\*^ | 75.4 | 83.5 | 92.4 | 90.5^\*\*\*^ | 58.3 | 80.4 |
| *Guidance* |  |  |  |  |  |  |  |
| MTL-Seg | 0.938^\*^ | 81.3 | 91.7 | 95.6 | 92.5 | 26.0^\*\*\*^ | 84.8 |
| RRR | 0.932 | 86.1 | 85.3 | 95.6^\*\*\*^ | 91.9 | 37.3^\*\*\*^ | 88.4 |
| CAM-Guide | 0.920 | 84.7 | 85.1 | 94.9 | 92.3 | 47.5^\*\*\*^ | 84.7 |
| CTR-Net | 0.933 | 79.8 | 91.5 | 94.4 | 93.4 | 3.3^\*\*\*^ | 86.4 |
| *Input restriction* |  |  |  |  |  |  |  |
| ROI-0.2 | 0.965^\*\*\*^ | 90.0 | 76.7 | 91.3 | 89.3^\*\*\*^ | 1.5^\*\*\*^ | 92.8 |
| ROI-0.6 | 0.966^\*\*\*^ | 94.1 | 80.0 | 94.2 | 90.8 | 1.8^\*\*\*^ | 92.5 |
| ROI-1.5 | 0.959^\*\*\*^ | 92.0 | 82.0 | 94.0 | 92.2 | 1.5^\*\*\*^ | 92.0 |
| ROI-Mask | 0.939 | 90.8 | 67.1 | 91.8 | 86.6^\*\*\*^ | 1.0^\*\*\*^ | 93.1 |
| ROI-Loc | 0.968^\*\*\*^ | 93.0 | 78.5 | 93.3 | 89.4^\*\*\*^ | 2.0^\*\*\*^ | 91.9 |
| *Tumor transplantation* |  |  |  |  |  |  |  |
| TT | 0.958^\*\*\*^ | 90.5 | 87.4 | 97.8 | 87.6^\*\*\*^ | 38.2^\*\*\*^ | 90.7 |
| *Both principles* |  |  |  |  |  |  |  |
| **ROI-0.6+TT** | **0.971^\*\*\*^** | **93.0** | **81.8** | **94.4** | **90.6** | **0.7^\*\*\*^** | **94.8** |
| *Skull control* |  |  |  |  |  |  |  |
| Baseline-Strip | 0.917 | 88.1 | 81.0 | 85.8^\*\*\*^ | 86.1^\*\*\*^ | 64.1^\*\*\*^ | 86.0 |
| *Foundation models* |  |  |  |  |  |  |  |
| DINOv2-S/14 | 0.749 | – | – | 67.4 | – | 62.0 | – |
| DINOv2-B/14 | 0.793 | – | – | 80.2 | – | 51.4 | – |
| BiomedCLIP | 0.770 | – | – | 85.7 | – | 89.6 | – |

Differences from the baseline after Holm correction: \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001 (BraTS and tumor free: patient bootstrap; BDNeuro and Mendeley: image bootstrap). – not computed.

**Table 5.** Explanation quality of two level LayerCAM maps (true class, mean of the five fold models). Columns 2 to 5 give the pointing score, the share of maps whose maximum lies inside the expert tumor mask. Energy ratio (internal test): share of map energy inside the mask divided by the share expected by chance within the region the model sees (Eq. 9). Faithfulness gap $G$ (Eq. 10): increase in the area under the deletion curve when random pixels instead of the most attributed pixels are removed, on 120 test slices.

| Model | Internal test | BRISC | BraTS glioma | BraTS meningioma | Energy ratio | Faithfulness gap $G$ (95% CI) |
|:------------|:--------|:--------|:--------|:-----------|:--------|:-----------------|
| Baseline | 0.36 | 0.59 | 0.46 | 0.29 | 3.66 | 0.18 (0.11 to 0.25) |
| MTL-Seg | 0.57 | 0.54 | 0.47 | 0.32 | 3.92 | – |
| CAM-Guide | 0.71 | 0.92 | 0.83 | 0.65 | 8.59 | 0.28 (0.22 to 0.34) |
| CTR-Net | 0.70 | 0.65 | 0.87 | 0.75 | 2.79 | 0.37 (0.31 to 0.43) |
| ROI-0.6 | 0.55 | 0.44 | 0.75 | 0.49 | 2.90 | – |
| ROI-Loc | 0.54 | 0.47 | 0.60 | 0.39 | 1.82 | 0.01 (−0.14 to 0.11) |
| TT | 0.56 | 0.74 | 0.49 | 0.63 | 6.44 | 0.59 (0.41 to 0.83) |
| **ROI-0.6+TT** | **0.86** | **0.87** | **0.87** | **0.79** | **4.15** | **−0.15 (−0.50 to 0.13)** |

Internal test: 479 slices; BRISC: 115 slices with expert masks that are not near duplicates of the development data; BraTS: 150 slices per class. – not computed.
