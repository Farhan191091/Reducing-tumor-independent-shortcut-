import numpy as np, cv2
from PIL import Image
D='/home/claude/figdata/figure_assets/'; A='assets/'; N=672
def up(n): return cv2.resize(np.asarray(Image.open(D+n).convert('L')),(N,N),interpolation=cv2.INTER_LANCZOS4)
def mk(n): return cv2.resize((np.asarray(Image.open(D+n).convert('L'))>127).astype(np.uint8),(N,N),interpolation=cv2.INTER_NEAREST)
def outline(img,mask,col,th=4,dash=False):
    rgb=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB); m=cv2.GaussianBlur(mask.astype(np.float32),(0,0),4)>0.5
    cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in cs:
        if dash:
            pts=c[:,0]; 
            for i in range(0,len(pts)-12,26): cv2.polylines(rgb,[pts[i:i+14]],False,col,th,cv2.LINE_AA)
        else: cv2.polylines(rgb,[c],True,col,th,cv2.LINE_AA)
    return rgb
for cls,t in [('g','inf_02'),('m','inf_10'),('p','inf_17')]:
    o=up(t+'_slice224.png'); em=mk(t+'_expertmask.png'); reg=mk(t+'_hide_region.png'); sp=mk(t+'_hide_spot.png')
    Image.fromarray(outline(o,em,(56,214,245))).save(A+f'h_{cls}_orig.png')
    Image.fromarray(outline(up(t+'_hide_swap.png'),reg,(251,113,133),3,True)).save(A+f'h_{cls}_swap.png')
    Image.fromarray(outline(up(t+'_hide_random.png'),sp,(203,213,225),3,True)).save(A+f'h_{cls}_rand.png')
    Image.fromarray(cv2.cvtColor(up(t+'_hide_skull.png'),cv2.COLOR_GRAY2RGB)).save(A+f'h_{cls}_skull.png')
    Image.open(D+t+'_native.png').convert('L').resize((N,N),Image.LANCZOS).save(A+f'cls_{cls}.png')
    Image.fromarray(up(t+'_hide_swap.png')).save(A+f'swapclean_{cls}.png')
print('ok')
