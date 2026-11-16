from common import *
import json, pandas as pd
P = json.load(open('assets/cam_probs.json'))
X = pd.read_csv('/home/claude/figdata/figure_assets/tables/table_xai_hires.csv')
MOD = [('CNX-Plain', 'Baseline', '#94a3b8'), ('CAM-Guide', 'CAM-Guide', VIOL), ('CNX-Transplant', 'TT', AMB), ('ROI-m06-Transplant', 'ROI-0.6+TT', ROSE)]
W, H = 1000, 590; f = Fig(W, H)
f.txt(4, 2, 30, 'a', 'letter'); f.txt(30, 6, 600, 'Attribution maps on held out test slices (true class)', 'ptitle')
BY = 36; BW = 640; f.box(0, BY, BW, H - BY, 'band')
S = 132; x0 = 64; y0 = BY + 52
for j, (_, name, c) in enumerate(MOD):
    x = x0 + j * (S + 12)
    f.txt(x - 6, BY + 16, S + 12, name, 't', f'color:{c};font-weight:700;font-size:14px;text-align:center;white-space:nowrap')
    for i, (cl, cn) in enumerate([('g', 'glioma'), ('m', 'meningioma'), ('p', 'pituitary')]):
        yy = y0 + i * (S + 26)
        f.im(f'cam_{cl}_{j}.png', x, yy, S)
        k = ['glioma', 'meningioma', 'pituitary'].index(cn); p = P[f'{cl}_{j}'][k]; ok = p == max(P[f'{cl}_{j}'])
        f.txt(x - 6, yy + S + 3, S + 12, f'<i>p</i>(true) = {p:.2f}' + ('' if ok else ' ✗'), 'capd', f'font-size:12.5px;{"color:#fda4af;font-weight:700" if not ok else ""}')
for i, cl in enumerate(['glioma', 'meningioma', 'pituitary']):
    yc = y0 + i * (S + 26) + S / 2
    f.els.append(f'<div class="rowl" style="left:{-44}px;top:{yc - 11}px;width:150px;color:{COLL[cl]}">{cl.capitalize()}</div>')
# legend in band bottom
ly = H - 26
f.els.append(f'<div class="lgd" style="left:{x0}px;top:{ly}px"><span style="display:inline-block;width:110px;height:10px;border-radius:3px;background-image:url({A}cmap.png);background-size:cover;vertical-align:0px;margin:0 6px"></span>low → high'
             f'<span style="margin-left:16px;color:{CY}">- - -</span> expert outline<span style="margin-left:14px;color:#fff">- - -</span> window seen by ROI-0.6+TT</div>')
# panel b: pointing score
PX = BW + 16; PW = W - PX
f.box(PX, BY, PW, H - BY, 'card', 'background:#fbfcfe')
f.txt(PX + 2, 2, 30, 'b', 'letter'); f.txt(PX + 28, 6, 330, 'Pointing score', 'ptitle')
f.txt(PX + 16, BY + 12, PW - 32, 'Share of maps whose maximum lies inside the expert tumor mask', 't', f'color:{MUT};font-size:12.5px;line-height:1.3')
DS = [('Internal test', 'Internal test (479)'), ('BRISC (dedup)', 'BRISC (115)'), ('BraTS-GLI', 'BraTS glioma (150)'), ('BraTS-MEN', 'BraTS meningioma (150)')]
gx0, gx1 = PX + 30, PX + PW - 26; sx = lambda v: gx0 + v * (gx1 - gx0)
top = BY + 82; gh = 98
for t in [0, .25, .5, .75, 1]:
    f.svg.append(f'<line x1="{sx(t)}" y1="{top - 8}" x2="{sx(t)}" y2="{top + 4 * gh - 16}" stroke="#e5e9f0"/>')
    f.txt(sx(t) - 20, top + 4 * gh - 12, 40, f'{t:g}', 'cap', f'font-size:12px;color:{MUT}')
DC = {'Baseline': '#475569', 'CAM-Guide': VIO, 'TT': '#c98a00', 'ROI-0.6+TT': '#e11d48'}
for d, (key, lab) in enumerate(DS):
    yy = top + d * gh
    f.txt(gx0 - 4, yy - 10, 260, lab, 't', 'font-size:13px;font-weight:700')
    for j, (m, name, _) in enumerate(MOD):
        v = float(X[(X.Dataset == key) & (X.Model == m)].pointing.iloc[0]); cy = yy + 18 + j * 16; c = DC[name]
        f.svg.append(f'<line x1="{sx(0)}" y1="{cy}" x2="{sx(v)}" y2="{cy}" stroke="{c}" stroke-opacity=".25" stroke-width="7" stroke-linecap="round"/>')
        f.svg.append(f'<circle cx="{sx(v)}" cy="{cy}" r="{5.5 if name == "ROI-0.6+TT" else 4.5}" fill="{c}" stroke="#fff" stroke-width="1.3"/>')
        f.txt(sx(v) + 8, cy - 8, 40, f'{v:.2f}', 't', f'font-size:11.5px;color:{c};font-weight:{800 if name == "ROI-0.6+TT" else 600}')
lx = PX + 16
for name, c in DC.items():
    f.els.append(f'<div class="lg" style="left:{lx}px;top:{H - 30}px"><i style="background:{c}"></i>{name}</div>'); lx += 22 + 7 * len(name) + 12
f.save('Figure7_v1', f'''.rowl{{font-size:14px;font-weight:700;text-align:center;transform:rotate(-90deg);white-space:nowrap}}
.lgd{{font-size:12.5px;color:#a9b8cd;white-space:nowrap;width:auto!important}}
.lg{{font-size:12.5px;white-space:nowrap;width:auto!important}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}}''')
