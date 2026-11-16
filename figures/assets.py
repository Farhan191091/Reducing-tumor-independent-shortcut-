import json, numpy as np, cv2, matplotlib
from PIL import Image
from scipy.ndimage import gaussian_filter
D='/home/claude/figdata/figure_assets/'; A='assets/'
INF,TT='inf_09','tt_06'
m={r['tag']:r for r in json.load(open(D+'inference_examples.json'))}[INF]
t={r['tag']:r for r in json.load(open(D+'transplant_examples.json'))}[TT]
def nat(n,s=720): return Image.open(D+n).convert('L').resize((s,s),Image.LANCZOS)
nat(INF+'_native.png').save(A+'slice.png'); nat(INF+'_crop_native.png',560).save(A+'crop.png')
p=np.asarray(Image.open(D+INF+'_probmap_fold1.png').convert('L').resize((720,720),Image.BICUBIC))/255.
c=matplotlib.colormaps['inferno'](0.30+0.70*p); c[...,3]=np.clip((p-0.08)*1.5,0,0.9)
Image.fromarray((c*255).astype(np.uint8)).save(A+'heat.png')
nat(TT+'_recipient_native.png').save(A+'recipient.png'); nat(TT+'_donor_native.png').save(A+'donor.png')
nat(TT+'_composite.png').save(A+'composite.png'); nat(TT+'_composite_crop224.png',560).save(A+'ccrop.png')
rm=np.asarray(Image.open(D+TT+'_recipient_mask.png'))>127; ys,xs=np.where(rm)
dn=np.asarray(Image.open(D+TT+'_donor_native.png').convert('L')); dm=np.asarray(Image.open(D+TT+'_donor_mask.png'))>127
x,y,w,h=t['donor_patch_xywh']; s=dn.shape[0]/224
al=cv2.GaussianBlur(cv2.dilate(dm.astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32),(7,7),2)
al=np.asarray(Image.fromarray((np.clip(al,0,1)*255).astype(np.uint8)).resize(dn.shape[::-1],Image.BICUBIC))
pa=np.dstack([dn,dn,dn,al])[int(y*s):int((y+h)*s),int(x*s):int((x+w)*s)]
Image.fromarray(pa).resize((pa.shape[1]*3,pa.shape[0]*3),Image.LANCZOS).save(A+'patch.png')
g={'probs':m['probs_5fold_mean'],'cls':m['class'],'win':[v/224 for v in m['window_fold1']],'box':[v/224 for v in m['box_fold1']],
   'rtum':[xs.min()/224,ys.min()/224,(np.ptp(xs)+1)/224,(np.ptp(ys)+1)/224],'don':[v/224 for v in t['donor_patch_xywh']],
   'placed':[v/224 for v in t['placed_xywh']],'cwin':[v/224 for v in t['window']],'rc':t['recipient_class'],'dc':t['donor_class'],'pa':pa.shape[1]/pa.shape[0]}
json.dump(g,open(A+'geom.json','w')); print(g)
