import os
A='file://'+os.path.abspath('assets')+'/'
INK='#0f172a'; MUT='#55637a'; TEAL='#2dd4bf'; TEALD='#0f8f83'; VIO='#6d43d8'; VIOL='#a78bfa'; AMB='#fbbf24'; CY='#38d6f5'; YEL='#ffc233'; ROSE='#fb7185'
COL={'glioma':'#0072B2','meningioma':'#D55E00','pituitary':'#009E73'}; COLL={'glioma':'#56b4e9','meningioma':'#ff8c42','pituitary':'#35d0a0'}
class Fig:
    def __init__(s,W,H): s.W,s.H=W,H; s.els=[]; s.svg=[]
    def box(s,x,y,w,h,cls='',style='',inner=''): s.els.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;{style}">{inner}</div>')
    def txt(s,x,y,w,t,cls='t',style=''): s.els.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;width:{w}px;{style}">{t}</div>')
    def im(s,src,x,y,sz,extra='',cls='imgd',style=''): s.els.append(f'<div class="{cls}" style="left:{x}px;top:{y}px;width:{sz}px;height:{sz}px;background-image:url({A}{src});{style}">{extra}</div>')
    def arrow(s,x0,y0,x1,y1,col,mk,w=2.6,glow=True): s.svg.append(f'<path d="M{x0},{y0} L{x1},{y1}" stroke="{col}" stroke-width="{w}" fill="none" marker-end="url(#{mk})" {"filter=\"url(#glow)\"" if glow else ""}/>')
    def save(s,name,extra_css=''):
        W,H=s.W,s.H
        css=f'''*{{box-sizing:border-box}} html,body{{margin:0;background:#fff}}
body{{width:{W}px;height:{H}px;position:relative;font-family:Inter,sans-serif;color:{INK};-webkit-font-smoothing:antialiased}}
body>div,body>img{{position:absolute}} svg{{position:absolute;left:0;top:0;pointer-events:none}}
.card{{border:1.2px solid #e2e8f0;border-radius:16px;background:#fff}}
.band{{border-radius:18px;background:radial-gradient(120% 140% at 30% 40%,#0b1526 0%,#04070d 62%,#020409 100%);box-shadow:inset 0 0 0 1px #1e293b}}
.letter{{font-size:22px;font-weight:800}} .ptitle{{font-size:16.5px;font-weight:700;letter-spacing:-.2px;white-space:nowrap}}
.imgd{{background-size:cover;border-radius:8px;overflow:hidden}}
.ring{{position:absolute;inset:0;border:2.4px solid {CY};border-radius:8px;box-shadow:0 0 16px rgba(56,214,245,.55)}}
.cap{{font-size:13px;line-height:1.3;color:#3f4b5e;text-align:center}} .capd{{font-size:13px;line-height:1.3;color:#a9b8cd;text-align:center}} .t{{font-size:13px;line-height:1.35}}
.num{{display:inline-block;width:20px;height:20px;line-height:20px;border-radius:50%;color:#042f2e;background:{TEAL};font-size:13px;font-weight:800;margin-right:6px;vertical-align:1px;text-align:center}}
.chip{{color:#fff;font-size:13px;font-weight:650;padding:2px 10px;border-radius:20px;white-space:nowrap;width:auto!important;height:auto!important}}
.formula{{background:#faf8ff;border:1.2px solid #d6c9f6;border-radius:12px;display:flex;align-items:center;justify-content:center;gap:24px;font-family:"Latin Modern Math","STIX Two Math",serif;font-size:19px;color:#2a1a66}}
{extra_css}'''
        mk=lambda i,c: f'<marker id="{i}" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 L2.2,4.5 Z" fill="{c}"/></marker>'
        defs=f'''<defs>{mk("aht",TEAL)}{mk("ahv",VIO)}{mk("ahr",ROSE)}{mk("ahy",YEL)}{mk("ahk","#475569")}
<filter id="glow" x="-50%" y="-200%" width="200%" height="500%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'''
        open(name+'.html','w').write(f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{"".join(s.els)}<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{defs}{"".join(s.svg)}</svg></body></html>')
        os.system(f'node render.js {os.path.abspath(name)}.html {os.path.abspath(name)} {W} {H}')
