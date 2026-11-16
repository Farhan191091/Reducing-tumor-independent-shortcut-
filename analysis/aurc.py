import numpy as np, pandas as pd
exec(open('analysis.py').read().split('rng_master = 42')[0])
def aurc(conf, err):
    o=np.argsort(-conf, kind='stable'); e=err[o]; return float(np.mean(np.cumsum(e)/np.arange(1,len(e)+1)))
rows=[]
for n in MODELS:
    r={'model':NAMES[n]}
    for s,y in [('test',yt),('brats',yb)]:
        P=ens(n,s,T_=TEMP); h=hard(P); conf=np.where(h<0,0,P.max(1)); err=(h!=y).astype(float)
        if s=='brats':
            m=np.isin(y,[0,1]); conf,err=conf[m],err[m]
        r[f'aurc_{s}']=100*aurc(conf,err)
    rows.append(r)
d=pd.DataFrame(rows); d.round(2).to_csv(str(O)+'/aurc.csv',index=False); print(d.round(2).to_string())
