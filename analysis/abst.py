import numpy as np, pandas as pd
exec(open('analysis.py').read().split('# ---------- temperatures')[0])
dev = sp.reset_index(drop=True); ydev=dev.label.values
def rel_probs(n):
    out={}; tidx=np.where(dev.split=='test')[0]
    for f in range(1,6):
        z=np.load(E/'reliance'/f'{n}_s0_f{f}.npz'); idx=np.concatenate([np.where(dev.split==f'fold{f}')[0],tidx]); nd=len(idx)-len(tidx)
        for c in z.files:
            out.setdefault(c,np.zeros((len(dev),K))); out[c][idx[:nd]]=z[c][:nd]; out[c][tidx]+=z[c][nd:]/5
    return out
rows=[]
for n in MODELS:
    R=rel_probs(n); ho,hh=hard(R['original']),hard(R['hidden (swap)'])
    ok=hh>=0
    # BA on non-abstained hidden slices only (conditional), original BA on same slices
    def ba(y,h): return np.mean([(h[y==k]==k).mean() for k in range(K)])
    bo=bal_acc(ydev,ho); bh_c=ba(ydev[ok],hh[ok]); bo_c=ba(ydev[ok],ho[ok])
    rho_c=np.clip((bo_c-bh_c)/(bo_c-1/3),0,1) if ok.sum()>0 else np.nan
    rows.append(dict(model=NAMES[n],abst_orig=100*(ho<0).mean(),abst_hidden=100*(~ok).mean(),rho_answered=rho_c,n_answered=int(ok.sum())))
d=pd.DataFrame(rows); d.round(3).to_csv(str(O)+'/abstention.csv',index=False); print(d.round(3).to_string()); print(list(R.keys()))
for n in ['ROI-Crop','ROI-Crop-m06','ROI-m06-Transplant','CNX-Transplant']:
    R=rel_probs(n); h=hard(R['skull / position only']); r=hard(R['random spot hidden'])
    print(n,'head-only abstain %.1f'%(100*(h<0).mean()),'randspot abstain %.1f'%(100*(r<0).mean()))
