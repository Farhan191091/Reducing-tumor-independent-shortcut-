import json, os, math
g=json.load(open('assets/geom.json')); A='file://'+os.path.abspath('assets')+'/'
W,H=1000,862
INK='#0f172a'; MUT='#55637a'; TEAL='#2dd4bf'; TEALD='#0f8f83'; VIO='#6d43d8'; VIOL='#a78bfa'; AMB='#fbbf24'; CY='#38d6f5'; YEL='#ffc233'
COL={'glioma':'#0072B2','meningioma':'#D55E00','pituitary':'#009E73'}; COLL={'glioma':'#56b4e9','meningioma':'#ff8c42','pituitary':'#35d0a0'}
els=[]; svg=[]
def box(x,y,w,h,cls='',style='',inner=''): els.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;{style}">{inner}</div>')
def txt(x,y,w,s,cls='t',style=''): els.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;width:{w}px;{style}">{s}</div>')
def im(src,x,y,s,extra='',cls='img'): els.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;width:{s}px;height:{s}px;background-image:url({A}{src});">{extra}</div>')
def rect(o,S,cls,extra=''):
    x,y,w,h=o; return f'<div class="{cls}" style="left:{x*S}px;top:{y*S}px;width:{w*S}px;height:{h*S}px;{extra}"></div>'
def arrow(x0,y0,x1,y1,col,mk,w=2.6,glow=True): svg.append(f'<path d="M{x0},{y0} L{x1},{y1}" stroke="{col}" stroke-width="{w}" fill="none" marker-end="url(#{mk})" {"filter=\"url(#glow)\"" if glow else ""}/>')
# ============ panel a : dark imaging band ============
txt(4,2,30,'a','letter'); txt(30,6,700,'Inference: the classifier sees only the tumor region','ptitle')
txt(W-300,8,296,'Proposed model: <b>ROI-0.6+TT</b>','tag','text-align:right')
BY=36; BH=340
box(0,BY,W,BH,'band')
S=170; Y=BY+50; X=[22,214,406]; cy=Y+S/2
def stage(n,label,cx): els.append(f'<div class="stage" style="left:{cx-100}px;top:{BY+16}px;width:200px;"><span class="num">{n}</span>{label}</div>')
stage(1,'Input slice',X[0]+S/2); stage(2,'Localize the tumor',X[1]+S/2); stage(3,'Select the region',X[2]+S/2)
wx0,wy0,wx1,wy1=g['win']; win=[wx0,wy0,wx1-wx0,wy1-wy0]
im('slice.png',X[0],Y,S,cls='imgd')
im('slice.png',X[1],Y,S,f'<div class="ov" style="background-image:url({A}heat.png)"></div>',cls='imgd')
im('slice.png',X[2],Y,S,f'<div class="dim"></div>'+rect(win,S,'roi',f'background-image:url({A}slice.png);background-size:{S}px {S}px;background-position:-{win[0]*S}px -{win[1]*S}px')+rect(g['box'],S,'tb'),cls='imgd')
txt(X[0]-10,Y+S+8,S+20,'Contrast enhanced T1 slice,<br>224 × 224 pixels','capd')
txt(X[1]-10,Y+S+8,S+20,'Predicted tumor<br>probability map','capd')
txt(X[2]-10,Y+S+8,S+20,'Largest component <i>C</i>,<br>window margin <i>m</i> = 0.6','capd')
for a,b in [(X[0]+S+4,X[1]-8),(X[1]+S+4,X[2]-8)]: arrow(a,cy,b,cy,TEAL,'aht')
bx,by,bs=X[2]+win[0]*S,Y+win[1]*S,win[2]*S
CX,CS=596,146; CYc=cy-CS/2
svg.append(f'<polygon points="{bx+bs},{by} {CX},{CYc} {CX},{CYc+CS} {bx+bs},{by+bs}" fill="url(#beam)"/>')
svg.append(f'<path d="M{bx+bs},{by} L{CX},{CYc} M{bx+bs},{by+bs} L{CX},{CYc+CS}" stroke="{CY}" stroke-width="1" stroke-opacity=".7"/>')
im('crop.png',CX,CYc,CS,'<div class="ring"></div>',cls='imgd')
txt(CX-14,Y+S+8,CS+28,'Region resized to<br>224 × 224 pixels','capd')
svg.append(f'<path d="M{bx+bs-3},{by} C{bx+bs+2},{Y-14} {bx+bs+10},{BY+28} {584},{BY+28}" stroke="{AMB}" stroke-width="1.8" fill="none" stroke-dasharray="5 4" marker-end="url(#aha)"/>')
txt(588,BY+13,150,'|<i>C</i>| &lt; 20 px: <b>abstain</b>','pilld')
stage(4,'Classify',822); arrow(CX+CS+5,cy,757,cy,TEAL,'aht')
xs=[764,790,820,854]; ts=[10,13,17,22]; hs=[112,88,64,44]; cols=['#7dd3fc','#38bdf8','#0ea5e9','#0284c7']; res=['56²','28²','14²','7²']; d=12
for x,t,h,c,r in zip(xs,ts,hs,cols,res):
    y0=cy-h/2
    svg.append(f'<g filter="url(#glow2)"><polygon points="{x+t},{y0} {x+t+d},{y0-d*.62} {x+t+d},{y0+h-d*.62} {x+t},{y0+h}" fill="{c}" opacity=".45"/><polygon points="{x},{y0} {x+d},{y0-d*.62} {x+t+d},{y0-d*.62} {x+t},{y0}" fill="{c}" opacity=".75"/><rect x="{x}" y="{y0}" width="{t}" height="{h}" fill="{c}"/></g>')
    txt(x-20,cy+60,t+40,r,'capd','text-align:center')
txt(737,Y+S+8,170,'ConvNeXt-T,<br>four stages','capd')
arrow(893,cy,904,cy,TEAL,'aht')
# output: hero number
best=max(range(3),key=lambda i:g['probs'][i]); names=['Glioma','Meningioma','Pituitary']
txt(908,BY+16,90,'<span class="num">5</span>Output','stage','text-align:left;left:910px')
for i,(n,v) in enumerate(zip(names,g['probs'])):
    c=COLL[n.lower()]; yy=Y+10+i*46
    txt(912,yy,86,n,'capd',f'color:{c};font-weight:650;text-align:left;font-size:13px')
    txt(912,yy+15,70,f'{v:.2f}','capd',f'color:#fff;font-weight:{"800" if i==best else "500"};font-size:{"22" if i==best else "15"}px;text-align:left;opacity:{1 if i==best else .75}')
txt(902,Y+S+8,96,'Mean of five<br>fold models','capd')
box(22,BY+BH-52,690,36,'stripd')
txt(36,BY+BH-43,262,'Hidden from the classifier by design:','t','font-weight:650;color:#99f6e4')
for s,x,w in [('skull and head shape',296,160),('slice level',464,96),('viewing plane',568,118)]: txt(x,BY+BH-47,w,s,'chipxd')
# ============ panel b : exploded layer view ============
PY=BY+BH+14; BW=600
box(0,PY,BW,H-PY,'card','background:linear-gradient(160deg,#f5f1ff 0%,#ffffff 55%);border-color:#ddd3f5')
txt(16,PY+10,30,'b','letter'); txt(42,PY+14,560,'Training: tumor transplantation across classes (<i>π</i> = 0.5)','ptitle',f'color:{VIO}')
T=150; a_=math.radians(56); b_=math.radians(-40)
def proj(cx,cy_,nx,ny):
    u,v=(nx-.5)*T,(ny-.5)*T
    return cx+u*math.cos(b_)-v*math.sin(b_), cy_+(u*math.sin(b_)+v*math.cos(b_))*math.cos(a_)
TX=124; DYc=PY+124; RYc=PY+286
def tile(src,cx,cy_,extra=''):
    els.append(f'<div class="shadow" style="left:{cx-95}px;top:{cy_+18}px;width:190px;height:70px"></div>')
    els.append(f'<div class="tile" style="left:{cx-T/2}px;top:{cy_-T/2}px;width:{T}px;height:{T}px;background-image:url({A}{src});">{extra}</div>')
tile('recipient_inp.png',TX,RYc,rect(g['placed'],T,'land'))
tile('donor.png',TX,DYc,rect(g['don'],T,'hole'))
dx,dy,dw,dh=g['don']; px,py,pw,ph=g['placed']
p1=proj(TX,DYc,dx+dw/2,dy+dh/2); p2=proj(TX,RYc,px+pw/2,py+ph/2)
LIFT=30; PWp=dw*T*1.25
els.append(f'<img src="{A}patch.png" class="patch3d" style="left:{p1[0]-PWp/2}px;top:{p1[1]-LIFT-PWp/2}px;width:{PWp}px">')
svg.append(f'<path d="M{p1[0]},{p1[1]-2} L{p1[0]},{p1[1]-LIFT+8}" stroke="{YEL}" stroke-width="1.4" stroke-dasharray="3 3"/>')
svg.append(f'<path d="M{p1[0]+10},{p1[1]-LIFT} C{p1[0]+78},{p1[1]-LIFT+10} {p2[0]+70},{p2[1]-70} {p2[0]+4},{p2[1]-5}" stroke="{YEL}" stroke-width="2.6" fill="none" marker-end="url(#ahy)"/>')
txt(p1[0]+76,(p1[1]+p2[1])/2-34,60,'<i>T</i>(<i>P<sub>d</sub></i>)','t',f'color:#9a6a00;font-weight:650;font-size:14px')
els.append(f'<div class="chip" style="left:16px;top:{DYc+70}px;background:{COL[g["dc"]]}">Donor: {g["dc"]}</div>')
els.append(f'<div class="chip" style="left:16px;top:{RYc+58}px;background:{COL[g["rc"]]}">Recipient: {g["rc"]}, tumor inpainted</div>')
# composite + crop
KX,KY,KS=292,PY+92,154
cw0,cw1,cw2,cw3=g['cwin']; cwin=[cw0,cw1,cw2-cw0,cw3-cw1]
arrow(236,KY+KS/2,KX-10,KY+KS/2,VIO,'ahv',3,False)
im('composite.png',KX,KY,KS,rect(cwin,KS,'roi')+f'<span class="lab" style="background:{COL[g["dc"]]};top:auto;bottom:6px"><i>y</i> = {g["dc"]}</span>')
KC=112; KCX,KCY=474,KY+(KS-KC)/2
kx,ky,ks=KX+cwin[0]*KS,KY+cwin[1]*KS,cwin[2]*KS
svg.append(f'<polygon points="{kx+ks},{ky} {KCX},{KCY} {KCX},{KCY+KC} {kx+ks},{ky+ks}" fill="url(#zoom)" stroke="{CY}" stroke-width="1" stroke-opacity=".6"/>')
im('ccrop.png',KCX,KCY,KC,'<div class="ring"></div>')
txt(KX-8,KY+KS+9,KS+16,'Composite <i>x</i><sup>TT</sup>: the label<br>follows the tumor','cap')
txt(KCX-12,KY+KS+9,KC+24,'Crop, random<br>margin and shift','cap')
txt(KX-8,KY+KS+58,BW-KX-6,'Skull, slice level and viewing plane now<br>come from another class, so only the<br>tumor predicts the label.','t',f'color:#4c2bb0;font-style:italic;line-height:1.38')
box(16,H-58,BW-32,42,'formula','','<span><i>x</i><sup>TT</sup> = <i>α</i> ⊙ <i>T</i>(<i>P<sub>d</sub></i>) + (1 − <i>α</i>) ⊙ <i>x<sub>r</sub></i><sup>inp</sup></span><span class="sep"></span><span><i>y</i><sup>TT</sup> = <i>y<sub>d</sub></i></span>')
# link b -> a
svg.append(f'<path d="M{KCX+KC/2+14},{KCY-4} C{KCX+KC/2+16},{PY-26} 806,{PY-2} 820,{Y+S+50}" stroke="{VIOL}" stroke-width="2" fill="none" stroke-dasharray="6 4" marker-end="url(#ahl)"/>')
txt(840,BY+BH-46,150,'trains this classifier','pill',f'color:#ddd6fe;background:rgba(167,139,250,.14);border-color:{VIOL};text-align:center;z-index:5')
# ============ panel c ============
CXp=BW+14
box(CXp,PY,W-CXp,H-PY,'card','border-color:#dfe5ec')
txt(CXp+16,PY+10,30,'c','letter'); txt(CXp+42,PY+14,330,'What each design lets the classifier use','ptitle')
rows=['Tumor appearance','Tumor location in head','Skull and head shape','Slice level','Viewing plane']
colsn=['Baseline','Input restriction','Transplantation','Combined']; thumbs=['slice.png','crop.png','composite.png','ccrop.png']
M=[[2,2,2,2],[2,0,2,0],[2,0,1,0],[2,0,1,0],[2,0,1,0]]
cw=52; gx=W-14-4*cw; TH=44; ty=PY+142; gy=ty+TH+14; rh=37
box(gx+3*cw-1,ty-8,cw,TH+16+5*rh+2,'hl')
for j,(n,th) in enumerate(zip(colsn,thumbs)):
    els.append(f'<div class="colh" style="left:{gx+j*cw+cw/2-2}px;top:{ty-32}px;{"font-weight:700;color:%s"%TEALD if j==3 else ""}">{n}</div>')
    els.append(f'<div class="thumb" style="left:{gx+j*cw+(cw-TH)/2}px;top:{ty}px;width:{TH}px;height:{TH}px;background-image:url({A}{th})"></div>')
txt(CXp+16,ty+13,150,'classifier input','t',f'color:{MUT};font-style:italic')
def dot(cx,cy_,v,r=8):
    if v==2: svg.append(f'<circle cx="{cx}" cy="{cy_}" r="{r}" fill="{INK}"/>')
    elif v==1: svg.append(f'<circle cx="{cx}" cy="{cy_}" r="{r-.8}" fill="#fff" stroke="{INK}" stroke-width="1.8"/><path d="M{cx},{cy_-r+.8} A{r-.8},{r-.8} 0 0 0 {cx},{cy_+r-.8} Z" fill="{INK}"/>')
    else: svg.append(f'<circle cx="{cx}" cy="{cy_}" r="{r-.8}" fill="#fff" stroke="#a9b4c2" stroke-width="1.6"/>')
for i,r in enumerate(rows):
    yy=gy+i*rh; txt(CXp+16,yy+9,166,r,'t','font-weight:500')
    svg.append(f'<line x1="{CXp+16}" y1="{yy}" x2="{W-12}" y2="{yy}" stroke="#eef1f5" stroke-width="1"/>')
    for j in range(4): dot(gx+j*cw+cw/2,yy+rh/2,M[i][j])
ly=gy+5*rh+16
for k,(v,s) in enumerate([(2,'visible and predictive of the label'),(1,'visible, link to label weakened'),(0,'not visible')]):
    dot(CXp+26,ly+k*23+9,v,6.4); txt(CXp+42,ly+k*23,300,s,'t',f'color:{MUT}')
css=f'''
*{{box-sizing:border-box}} html,body{{margin:0;background:#fff}}
body{{width:{W}px;height:{H}px;position:relative;font-family:Inter,sans-serif;color:{INK};-webkit-font-smoothing:antialiased}}
body>div,body>img{{position:absolute}} svg{{position:absolute;left:0;top:0;pointer-events:none}}
.card{{border:1.2px solid #e2e8f0;border-radius:16px;background:#fff}}
.band{{border-radius:18px;background:radial-gradient(120% 140% at 30% 40%,#0b1526 0%,#04070d 62%,#020409 100%);box-shadow:inset 0 0 0 1px #1e293b}}
.letter{{font-size:22px;font-weight:800}} .ptitle{{font-size:16.5px;font-weight:700;letter-spacing:-.2px;white-space:nowrap}}
.tag{{font-size:13px;color:{MUT}}} .tag b{{color:{INK}}}
.stage{{font-size:14px;font-weight:650;text-align:center;white-space:nowrap;color:#f1f5f9}}
.num{{display:inline-block;width:20px;height:20px;line-height:20px;border-radius:50%;color:#042f2e;background:{TEAL};font-size:13px;font-weight:800;margin-right:6px;vertical-align:1px;text-align:center}}
.img{{background-size:cover;border-radius:10px;box-shadow:0 10px 22px -8px rgba(15,23,42,.5),0 0 0 1px rgba(15,23,42,.6);overflow:hidden}}
.imgd{{background-size:cover;border-radius:8px;overflow:hidden}}
.ov{{position:absolute;inset:0;background-size:cover}} .dim{{position:absolute;inset:0;background:rgba(2,6,12,.62)}}
.roi{{position:absolute;border:2.4px solid {CY};border-radius:3px;box-shadow:0 0 0 1px rgba(0,0,0,.4),0 0 14px rgba(56,214,245,.75)}}
.tb{{position:absolute;border:1.5px dashed #fff;border-radius:2px}}
.ring{{position:absolute;inset:0;border:2.4px solid {CY};border-radius:8px;box-shadow:0 0 16px rgba(56,214,245,.55)}}
.lab{{position:absolute;left:6px;top:6px;color:#fff;font-size:13px;font-weight:650;padding:1px 8px;border-radius:20px;box-shadow:0 1px 3px rgba(0,0,0,.45);white-space:nowrap}}
.chip{{color:#fff;font-size:13px;font-weight:650;padding:2px 10px;border-radius:20px;white-space:nowrap;width:auto!important;height:auto!important;box-shadow:0 2px 5px rgba(15,23,42,.25)}}
.cap{{font-size:13px;line-height:1.3;color:#3f4b5e;text-align:center}} .capd{{font-size:13px;line-height:1.3;color:#a9b8cd;text-align:center}} .t{{font-size:13px}}
.pill{{font-size:13px;padding:5px 11px;border:1.2px solid;border-radius:20px;white-space:nowrap;width:auto!important}}
.pilld{{font-size:13px;padding:5px 11px;border:1.2px solid {AMB};border-radius:20px;white-space:nowrap;width:auto!important;color:#fde68a;background:rgba(251,191,36,.12)}}
.stripd{{background:rgba(45,212,191,.09);border-radius:12px;border:1px solid rgba(45,212,191,.35)}}
.chipxd{{font-size:13px;padding:4px 0;background:rgba(255,255,255,.06);border:1px solid rgba(148,163,184,.45);border-radius:20px;color:#cbd5e1;text-decoration:line-through;text-decoration-color:#fb7185;text-decoration-thickness:1.8px;text-align:center;white-space:nowrap}}
.tile{{background-size:cover;border-radius:6px;transform:rotateX(56deg) rotateZ(-40deg);box-shadow:0 0 0 1.5px #1e293b, -5px 5px 0 0 #334155, -9px 9px 0 0 #1e293b;overflow:hidden}}
.shadow{{background:radial-gradient(closest-side,rgba(30,20,80,.32),rgba(30,20,80,0));border-radius:50%;filter:blur(4px)}}
.hole{{position:absolute;border:2px solid {YEL};border-radius:4px;background:rgba(255,194,51,.22)}}
.land{{position:absolute;border:2px dashed {YEL};border-radius:4px;background:rgba(255,194,51,.18)}}
.patch3d{{transform:rotateX(56deg) rotateZ(-40deg);filter:drop-shadow(0 0 3px {YEL}) drop-shadow(0 6px 6px rgba(0,0,0,.45));z-index:6}}
.formula{{background:#faf8ff;border:1.2px solid #d6c9f6;border-radius:12px;display:flex;align-items:center;justify-content:center;gap:24px;font-family:"Latin Modern Math","STIX Two Math",serif;font-size:19px;color:#2a1a66}}
.formula .sep{{width:1px;height:24px;background:#cdbff2}}
.hl{{background:#e6f5f2;border:1px solid #bfe3db;border-radius:10px}}
.thumb{{background-size:cover;border-radius:6px;box-shadow:0 0 0 1px #1e293b,0 3px 6px rgba(15,23,42,.3)}}
.colh{{font-size:13px;white-space:nowrap;transform-origin:left bottom;transform:rotate(-50deg);color:#334155;width:auto!important}}
'''
mk=lambda i,c: f'<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 L2.2,4.5 Z" fill="{c}"/></marker>'
defs=f'''<defs>{mk("aht",TEAL)}{mk("ahv",VIO)}{mk("ahl",VIOL)}{mk("aha",AMB)}{mk("ahy",YEL)}
<linearGradient id="zoom" x1="0" x2="1"><stop offset="0" stop-color="{CY}" stop-opacity=".32"/><stop offset="1" stop-color="{CY}" stop-opacity=".07"/></linearGradient>
<linearGradient id="beam" x1="0" x2="1"><stop offset="0" stop-color="{CY}" stop-opacity=".42"/><stop offset="1" stop-color="{CY}" stop-opacity=".10"/></linearGradient>
<filter id="glow" x="-50%" y="-200%" width="200%" height="500%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<filter id="glow2" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'''
open('fig_v3.html','w').write(f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{"".join(els)}<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{defs}{"".join(svg)}</svg></body></html>')
print(W,H)
