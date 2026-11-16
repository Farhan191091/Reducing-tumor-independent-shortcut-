from common import *
W,H=1000,412; f=Fig(W,H)
CW,G=238,16; BY=0; BH=200
f.box(0,BY,W,BH,'band')
heads=[('a','Data'),('b','Diagnose the shortcut'),('c','Proposed framework'),('d','Compare and evaluate')]
X=[i*(CW+G) for i in range(4)]
for (l,t),x in zip(heads,X):
    f.txt(x+14,BY+12,CW,f'<span class="lt">{l}</span>{t}','hd')
for x in X[1:]:
    f.svg.append(f'<path d="M{x-G/2-5},{BY+BH/2-12} l9,12 l-9,12" stroke="{TEAL}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>')
IY=BY+50
# a: three classes
for i,(c,n) in enumerate([('g','glioma'),('m','meningioma'),('p','pituitary')]):
    x=X[0]+4+i*78; f.im(f'cls_{c}.png',x,IY+6,70)
    f.txt(x-4,IY+82,78,n.capitalize(),'capd',f'color:{COLL[n]};font-weight:600;font-size:13px')
f.txt(X[0]+8,IY+112,CW-16,'Figshare, contrast enhanced T1','capd')
# b: original -> hidden
f.im('h_m_orig.png',X[1]+10,IY+4,100); f.im('h_m_swap.png',X[1]+128,IY+4,100)
f.arrow(X[1]+112,IY+54,X[1]+125,IY+54,ROSE,'ahr',2.4)
f.txt(X[1]+10,IY+108,100,'original','capd'); f.txt(X[1]+128,IY+108,100,'tumor hidden','capd',f'color:{ROSE}')
# c: crop + composite
f.im('crop.png',X[2]+10,IY+4,100,'<div class="ring"></div>'); f.im('composite.png',X[2]+128,IY+4,100)
f.txt(X[2]+4,IY+108,112,'tumor region only','capd',f'color:{CY}'); f.txt(X[2]+116,IY+108,124,'transplanted tumor','capd',f'color:{YEL};white-space:nowrap')
f.svg.append(f'<circle cx="{X[2]+128+0.59*100}" cy="{IY+4+0.25*100}" r="13" fill="none" stroke="{YEL}" stroke-width="2" stroke-dasharray="4 3"/>')
# d: heat map + bars glyph
f.im('slice.png',X[3]+10,IY+4,100,f'<div style="position:absolute;inset:0;background-size:cover;background-image:url({A}heat.png)"></div>')
gx=X[3]+132; gb=IY+100
for k,(h,c) in enumerate([(38,'#475569'),(52,'#64748b'),(60,'#94a3b8'),(88,TEAL)]):
    f.svg.append(f'<rect x="{gx+k*23}" y="{gb-h}" width="16" height="{h}" rx="4" fill="{c}" {"filter=\"url(#glow)\"" if k==3 else ""}/>')
f.svg.append(f'<line x1="{gx-6}" y1="{gb+1}" x2="{gx+92}" y2="{gb+1}" stroke="#64748b" stroke-width="1.2"/>')
f.txt(X[3]+4,IY+108,112,'explanations','capd'); f.txt(X[3]+122,IY+108,112,'benchmarks','capd')
# text cards
TY=BY+BH+12; TH=H-TY
cards=[
 ['<b>3,064</b> slices, <b>233</b> patients','Development: 2,585 slices, 198 patients, fivefold cross validation','Held out test: 479 slices, 35 patients','External, duplicates removed: BraTS 1,073; tumor free 911; BDNeuro 227; Mendeley 1,108'],
 ['Hide the tumor, keep everything else','Tumor reliance score <i>ρ</i>: 0 = tumor not needed, 1 = tumor required','Controls: random spot hidden, head geometry only'],
 ['<b>Input restriction:</b> classify only the detected tumor region','<b>Tumor transplantation:</b> train with tumors placed in slices of another class','Combined model: <b>ROI-0.6+TT</b>'],
 ['Against the baseline, standard CNNs and transformers, four guidance methods and frozen foundation models','Cross validation, held out test, external transfer','Tumor free safety test, referral, explanation faithfulness'],
]
accent=['#0072B2','#e11d48',TEALD,'#b7791f']
for x,items,c in zip(X,cards,accent):
    f.box(x,TY,CW,TH,'card',f'border-top:4px solid {c}')
    f.els.append(f'<ul class="ul" style="left:{x}px;top:{TY+10}px;width:{CW}px">'+''.join(f'<li style="--c:{c}">{t}</li>' for t in items)+'</ul>')
f.save('Figure1_v3',f'''.hd{{font-size:15px;font-weight:700;color:#f1f5f9;white-space:nowrap}} .lt{{font-size:20px;font-weight:800;margin-right:9px;color:{TEAL}}}
.ul{{position:absolute;margin:0;padding:0 14px 0 30px;list-style:none;font-size:13px;line-height:1.36;color:#1e293b}} .ul li{{position:relative;margin:0 0 9px}} .ul li:before{{content:"";position:absolute;left:-15px;top:6px;width:6px;height:6px;border-radius:50%;background:var(--c)}}''')
