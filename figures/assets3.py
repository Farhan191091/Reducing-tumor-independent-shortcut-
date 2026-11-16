import numpy as np, cv2, matplotlib
from PIL import Image
D='/home/claude/figdata/figure_assets/'; A='assets/'; N=672
H=np.load('/home/claude/rexp/results_export/heatmaps_24.npz')
stems=list(H['stems']); cmap=matplotlib.colormaps['inferno']
MODELS=['CNX-Plain','CAM-Guide','CNX-Transplant','ROI-m06-Transplant']
info={}
for cls,t,stem in [('g','inf_02','g_MR051644_s_023'),('m','inf_10','m_97943_s_003'),('p','inf_17','p_103671_s_008')]:
    i=stems.index(stem); bg=np.asarray(Image.open(D+t+'_native.png').convert('L').resize((N,N),Image.LANCZOS)).astype(np.float32)/255
    em=cv2.resize((np.asarray(Image.open(D+t+'_expertmask.png').convert('L'))>127).astype(np.uint8),(N,N),interpolation=cv2.INTER_NEAREST)
    em=cv2.GaussianBlur(em.astype(np.float32),(0,0),4)>0.5
    cs,_=cv2.findContours(em.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for mi,m in enumerate(MODELS):
        cam=H[f'{m}__cam'][i].astype(np.float32); win=None
        if m.startswith('ROI'):
            ys,xs=np.where(cam>0); win=(xs.min(),ys.min(),xs.max()+1,ys.max()+1)
        sm=cv2.GaussianBlur(cam,(0,0),6); sm=sm/(sm.max()+1e-8)
        sm=cv2.resize(sm,(N,N),interpolation=cv2.INTER_CUBIC).clip(0,1)
        rgb=np.repeat(bg[...,None],3,2)*0.92
        a=(sm**1.2)[...,None]*0.82; col=cmap(sm)[...,:3]
        out=(rgb*(1-a)+col*a); out=(out*255).astype(np.uint8).copy()
        for c in cs:
            pts=c[:,0]
            for k in range(0,len(pts)-10,22): cv2.polylines(out,[pts[k:k+12]],False,(56,214,245),3,cv2.LINE_AA)
        if win is not None:
            s=N/224; x0,y0,x1,y1=[int(v*s) for v in win]
            for (p,q) in [((x0,y0),(x1,y0)),((x1,y0),(x1,y1)),((x1,y1),(x0,y1)),((x0,y1),(x0,y0))]:
                L=int(np.hypot(q[0]-p[0],q[1]-p[1])); 
                for k in range(0,L,24):
                    a0=np.array(p)+(np.array(q)-np.array(p))*k/L; a1=np.array(p)+(np.array(q)-np.array(p))*min(k+13,L)/L
                    cv2.line(out,tuple(a0.astype(int)),tuple(a1.astype(int)),(255,255,255),3,cv2.LINE_AA)
        Image.fromarray(out).save(A+f'cam_{cls}_{mi}.png')
        info[f'{cls}_{mi}']=[float(v) for v in H[f'{m}__probs'][i]]
import json; json.dump(info,open(A+'cam_probs.json','w'),indent=1); print(info)
# colorbar
cb=(cmap(np.linspace(0,1,512))[None,:,:3]*255).astype(np.uint8).repeat(24,0); Image.fromarray(cb).save(A+'cmap.png')
