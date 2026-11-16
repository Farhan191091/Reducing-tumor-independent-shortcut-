import pandas as pd, numpy as np
import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O=ROOT+'/results/'; T=ROOT+'/results/run6_tables/'
M=pd.read_csv(O+'main.csv').merge(pd.read_csv(O+'abstention.csv'),on='model').set_index('model'); E=pd.read_csv(O+'external.csv'); C=pd.read_csv(O+'comparisons.csv')
INV={'CNX-Plain':'Baseline','CAM-Guide':'CAM-Guide','CNX-Transplant':'TT','CTR-Net':'CTR-Net','MTL-Seg':'MTL-Seg','ROI-Crop-m06':'ROI-0.6','ROI-Loc':'ROI-Loc','ROI-m06-Transplant':'ROI-0.6+TT'}
GROUPS=[('Standard',['Baseline','ConvNeXt-T (standard head)','ResNet-50','EfficientNet-B0','DeiT-S']),('Guidance',['MTL-Seg','RRR','CAM-Guide','CTR-Net']),
 ('Input restriction',['ROI-0.2','ROI-0.6','ROI-1.5','ROI-Mask','ROI-Loc']),('Tumor transplantation',['TT']),('Both principles',['ROI-0.6+TT']),('Skull control',['Baseline-Strip'])]
def star(model,metric):
    if model=='Baseline': return ''
    r=C[(C.family=='vs_baseline')&(C.metric==metric)&(C.a==model)]
    if r.empty: return ''
    p=r.p_holm.iloc[0]; return '\\*\\*\\*' if p<.001 else '\\*\\*' if p<.01 else '\\*' if p<.05 else ''
def sup(s): return f'^{s}^' if s else ''
bd=E[E.set=='bdneuro'].set_index('model'); me=E[E.set=='mendeley'].set_index('model')
cf=pd.read_csv(T+'table_conformal.csv'); cf=cf[(cf.alpha==.05)&(cf.Set=='BraTS')]
NM={'CNX-Plain':'Baseline','ResNet-50':'ResNet-50','EfficientNet-B0':'EfficientNet-B0','ConvNeXt-Tiny':'ConvNeXt-T (standard head)','DeiT-Small':'DeiT-S','MTL-Seg':'MTL-Seg','RRR':'RRR','CAM-Guide':'CAM-Guide','CTR-Net':'CTR-Net','ROI-Crop':'ROI-0.2','ROI-Crop-m06':'ROI-0.6','ROI-Crop-m15':'ROI-1.5','ROI-Mask':'ROI-Mask','ROI-Loc':'ROI-Loc','CNX-Transplant':'TT','ROI-m06-Transplant':'ROI-0.6+TT','CNX-Strip':'Baseline-Strip'}
COV={NM[m]:v for m,v in zip(cf.Model,cf['Coverage (%)']) if m in NM}
FM=pd.read_csv(T+'table_foundation_models.csv'); FMN={'DINOv2-S':'DINOv2-S/14','DINOv2-B':'DINOv2-B/14','BiomedCLIP':'BiomedCLIP'}
def B(m,s): return f'**{s}**' if m=='ROI-0.6+TT' else s
out=[]
# ---------- Table 3 ----------
out.append('**Table 3.** Internal accuracy and tumor reliance. Cross validation (CV) macro F1 is the mean ± standard deviation over all runs; test results use the five fold ensemble of seed 0 on the 479 test slices (35 patients). Tumor reliance $\\rho$ (Eq. 3) uses the swap operator on all 3,064 slices with the first seed; abstained: share of hidden slices on which the model abstained; patients: patients correctly classified by majority vote; $\\rho_{\\mathrm{ans}}$: reliance on the answered slices only; head only: balanced accuracy when only the head outline remains (chance 33.3%). BA, balanced accuracy; CI, patient bootstrap confidence interval.\n')
out.append('| Model | CV macro F1 (%) | Test macro F1 (%) | Patients | $\\rho$ (95% CI) | Abstained (%) | $\\rho_{\\mathrm{ans}}$ | Head only BA (%) |')
out.append('|:------------|:----------|:-------|:---------|:----------------------|:----------|:-----|:-------|')
for g,ms in GROUPS:
    for i,m in enumerate(ms):
        r=M.loc[m]
        if i==0: out.append(f'| *{g}* |  |  |  |  |  |  |  |')
        hd=f"{r.ba_head_only:.1f}"+("^b^" if r.abst_hidden>0 else "")
        cells=[m,f"{r.cv_f1_mean:.1f} ± {r.cv_f1_sd:.1f}{sup(star(m,'cv_f1'))}",f"{r.test_f1:.1f}{sup(star(m,'test_f1'))}",
               f"{int(r.patients_correct)}/35",f"{r.rho:.2f} ({r.rho_lo:.2f}–{r.rho_hi:.2f}){sup(star(m,'rho'))}",f"{r.abst_hidden:.1f}",f"{r.rho_answered:.2f}",hd]
        out.append('| '+' | '.join(B(m,c) for c in cells)+' |')
for i,(k,n) in enumerate(FMN.items()):
    r=FM[FM.Model==k].iloc[0]
    if i==0: out.append('| *Foundation models^a^* |  |  |  |  |  |  |  |')
    out.append(f"| {n} | – | – | – | {r['Tumour reliance']:.2f} | 0.0 | {r['Tumour reliance']:.2f} | {r['skull / position only']:.1f} |")
out.append('\nDifferences from the baseline after Holm correction: \\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001 (CV: corrected resampled t test; test and $\\rho$: patient bootstrap). ^a^ Frozen encoders with a logistic regression classifier, evaluated without cross validation (internal test balanced accuracy 83.2% to 84.8%); macro F1 and patient votes were not computed. ^b^ The model abstained on 98.6% of head only slices, so this value reflects abstention.\n')
# ---------- Table 4 ----------
out.append('**Table 4.** Transfer and safety. BraTS: 1,073 slices of 368 patients; AUC for glioma versus meningioma; Recall G and Recall M, recall of glioma and meningioma. BDNeuro (227 slices) and Mendeley (1,108 slices): balanced accuracy on the images that are not near duplicates of the development data. Tumor free: share of the 911 BraTS tumor free slices answered with a calibrated maximum probability of at least 0.85. Coverage: empirical coverage of cross conformal prediction sets on BraTS at a target of 95%. All values except AUC are percentages. BA, balanced accuracy.\n')
out.append('| Model | BraTS AUC | Recall G | Recall M | BDNeuro | Mendeley | Tumor free | Coverage |')
out.append('|:-------------------|:---------|:---------|:---------|:----------|:----------|:----------|:----------|')
for g,ms in GROUPS:
    for i,m in enumerate(ms):
        r=M.loc[m]
        if i==0: out.append(f'| *{g}* |  |  |  |  |  |  |  |')
        cells=[m,f"{r.brats_auc:.3f}{sup(star(m,'brats_auc'))}",f"{r.gli_recall:.1f}",f"{r.men_recall:.1f}",
               f"{bd.loc[m,'ba']:.1f}{sup(star(m,'bdneuro_ba'))}",f"{me.loc[m,'ba']:.1f}{sup(star(m,'mendeley_ba'))}",f"{r.free_confident:.1f}{sup(star(m,'free_confident'))}",f"{COV.get(m,float('nan')):.1f}"]
        out.append('| '+' | '.join(B(m,c) for c in cells)+' |')
for i,(k,n) in enumerate(FMN.items()):
    r=FM[FM.Model==k].iloc[0]
    if i==0: out.append('| *Foundation models* |  |  |  |  |  |  |  |')
    out.append(f"| {n} | {r['BraTS AUC']:.3f} | – | – | {r['BDNeuro bal. acc (%)']:.1f} | – | {r['Confident on tumour-free (%)']:.1f} | – |")
out.append('\nDifferences from the baseline after Holm correction: \\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001 (BraTS and tumor free: patient bootstrap; BDNeuro and Mendeley: image bootstrap). – not computed.\n')
# ---------- Table 5 ----------
X=pd.read_csv(T+'table_xai_hires.csv'); FA=pd.read_csv(T+'table_xai_faithfulness.csv').set_index('Model')
out.append('**Table 5.** Explanation quality of two level LayerCAM maps (true class, mean of the five fold models). Columns 2 to 5 give the pointing score, the share of maps whose maximum lies inside the expert tumor mask. Energy ratio (internal test): share of map energy inside the mask divided by the share expected by chance within the region the model sees (Eq. 9). Faithfulness gap $G$ (Eq. 10): increase in the area under the deletion curve when random pixels instead of the most attributed pixels are removed, on 120 test slices.\n')
out.append('| Model | Internal test | BRISC | BraTS glioma | BraTS meningioma | Energy ratio | Faithfulness gap $G$ (95% CI) |')
out.append('|:------------|:--------|:--------|:--------|:-----------|:--------|:-----------------|')
for k in ['CNX-Plain','MTL-Seg','CAM-Guide','CTR-Net','ROI-Crop-m06','ROI-Loc','CNX-Transplant','ROI-m06-Transplant']:
    m=INV[k]; pt=lambda d: X[(X.Dataset==d)&(X.Model==k)].pointing.iloc[0]
    er=X[(X.Dataset=='Internal test')&(X.Model==k)]['heat in tumour / chance'].iloc[0]
    g=f"{FA.loc[k,'faithfulness gap (random - red)']:.2f} {FA.loc[k,'95% CI'].replace('[','[').replace(', ',', ')}" if k in FA.index else '–'
    if k in FA.index:
        lo,hi=[float(v) for v in FA.loc[k,'95% CI'].strip('[]').split(',')]; g=f"{FA.loc[k,'faithfulness gap (random - red)']:.2f} ({lo:.2f} to {hi:.2f})"
    cells=[m,f'{pt("Internal test"):.2f}',f'{pt("BRISC (dedup)"):.2f}',f'{pt("BraTS-GLI"):.2f}',f'{pt("BraTS-MEN"):.2f}',f'{er:.2f}',g]
    out.append('| '+' | '.join(B(m,c) for c in cells)+' |')
out.append('\nInternal test: 479 slices; BRISC: 115 slices with expert masks that are not near duplicates of the development data; BraTS: 150 slices per class. – not computed.\n')
txt='\n'.join(out); import re; txt=re.sub(r'(?<=[\s\[(*])-(?=\d)','−',txt); open(O+'tables_3_to_5.md','w').write(txt)
