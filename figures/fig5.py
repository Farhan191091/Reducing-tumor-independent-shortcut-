from common import *
import pandas as pd
from scipy.stats import spearmanr
M = pd.read_csv('/home/claude/results/out/main.csv').set_index('model')
GC = {'Standard': '#475569', 'Guidance': VIO, 'Input restriction': TEALD, 'Transplantation': '#c98a00', 'Both principles': '#e11d48', 'Skull control': '#94a3b8'}
GROUP = {**{m: 'Standard' for m in ['Baseline', 'ConvNeXt-T (standard head)', 'ResNet-50', 'EfficientNet-B0', 'DeiT-S']},
         **{m: 'Guidance' for m in ['MTL-Seg', 'RRR', 'CAM-Guide', 'CTR-Net']},
         **{m: 'Input restriction' for m in ['ROI-0.2', 'ROI-0.6', 'ROI-1.5', 'ROI-Mask', 'ROI-Loc']},
         'TT': 'Transplantation', 'ROI-0.6+TT': 'Both principles', 'Baseline-Strip': 'Skull control'}
W, H = 1000, 600; f = Fig(W, H)
PAN = [dict(L='a', title='Reliance predicts transfer to BraTS', col='brats_auc', lo=.72, hi=1.0, ticks=[.75, .8, .85, .9, .95, 1.0], ylab='BraTS glioma vs meningioma AUC', fmt='{:.2f}', ci=True,
            lab={'Baseline': (230, 168), 'ResNet-50': (141, 300), 'EfficientNet-B0': (131, 393), 'DeiT-S': (172, 241), 'ROI-Mask': (396, 158),
                 'ROI-0.2': (400, 120), 'ROI-0.6': (250, 70), 'ROI-1.5': (240, 128), 'ROI-Loc': (400, 96), 'TT': (305, 150), 'ROI-0.6+TT': (395, 72),
                 'Baseline-Strip': (215, 222), '_guid': (38, 190)}),
       dict(L='b', title='Reliance costs some internal accuracy', col='cv_f1_mean', lo=86, hi=99, ticks=[87, 90, 93, 96, 99], ylab='Cross validation macro F1 (%)', fmt='{:g}', ci=False,
            lab={'Baseline': (750, 102), 'ResNet-50': (652, 270), 'EfficientNet-B0': (652, 206), 'DeiT-S': (668, 162), 'ROI-Mask': (888, 451),
                 'ROI-0.2': (848, 262), 'ROI-0.6': (790, 172), 'ROI-1.5': (765, 248), 'ROI-Loc': (870, 222), 'TT': (830, 162), 'ROI-0.6+TT': (866, 300),
                 'Baseline-Strip': (735, 150), '_guid': (600, 90)})]
CW = 484; GX = [0, W - CW]; TOP = 40
for P, gx in zip(PAN, GX):
    f.box(gx, TOP, CW, H - TOP - 50, 'card', 'background:#fbfcfe')
    f.txt(gx + 2, 2, 30, P['L'], 'letter'); f.txt(gx + 28, 6, CW, P['title'], 'ptitle', 'font-size:15.5px')
    x0, x1, y0, y1 = gx + 70, gx + CW - 22, TOP + 22, H - 50 - 52
    sx = lambda v: x0 + v * (x1 - x0)
    sy = lambda v, P=P: y1 - (v - P['lo']) / (P['hi'] - P['lo']) * (y1 - y0)
    for t in [0, .25, .5, .75, 1]:
        f.svg.append(f'<line x1="{sx(t)}" y1="{y0}" x2="{sx(t)}" y2="{y1}" stroke="#e5e9f0"/>'); f.txt(sx(t) - 25, y1 + 6, 50, f'{t:g}', 'cap', f'font-size:12px;color:{MUT}')
    for t in P['ticks']:
        f.svg.append(f'<line x1="{x0}" y1="{sy(t)}" x2="{x1}" y2="{sy(t)}" stroke="#e5e9f0"/>'); f.txt(x0 - 48, sy(t) - 8, 40, P['fmt'].format(t), 't', f'font-size:12px;color:{MUT};text-align:right')
    f.txt(x0, y1 + 24, x1 - x0, 'Tumor reliance <i>ρ</i>', 'cap', f'color:{MUT};font-size:12.5px')
    f.txt(gx - 100, (y0 + y1) / 2 - 8, 220, P['ylab'], 'cap', f'color:{MUT};font-size:12.5px;transform:rotate(-90deg);white-space:nowrap')
    # context sweep path and combination arrows
    pts = [(M.loc[m, 'rho'], M.loc[m, P['col']]) for m in ['ROI-0.2', 'ROI-0.6', 'ROI-1.5']]
    f.svg.append('<polyline points="' + ' '.join(f'{sx(a)},{sy(b)}' for a, b in pts) + f'" fill="none" stroke="{TEALD}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    for src in ['ROI-0.6', 'TT']:
        a0, b0, a1, b1 = sx(M.loc[src, 'rho']), sy(M.loc[src, P['col']]), sx(M.loc['ROI-0.6+TT', 'rho']), sy(M.loc['ROI-0.6+TT', P['col']])
        k = 9 / max(((a1 - a0) ** 2 + (b1 - b0) ** 2) ** .5, 1)
        f.svg.append(f'<path d="M{a0 + (a1 - a0) * k},{b0 + (b1 - b0) * k} L{a1 - (a1 - a0) * k * 1.1},{b1 - (b1 - b0) * k * 1.1}" stroke="#e11d48" stroke-opacity=".55" stroke-width="1.6" fill="none" marker-end="url(#ahr)"/>')
    for m in M.index:
        c = GC[GROUP[m]]; X, Y = sx(M.loc[m, 'rho']), sy(M.loc[m, P['col']]); prop = m == 'ROI-0.6+TT'
        if P['ci']:
            f.svg.append(f'<line x1="{X}" y1="{sy(M.loc[m, "brats_auc_lo"])}" x2="{X}" y2="{sy(M.loc[m, "brats_auc_hi"])}" stroke="{c}" stroke-opacity=".45" stroke-width="2"/>')
        f.svg.append(f'<circle cx="{X}" cy="{Y}" r="{7 if prop else 5.5}" fill="{c}" stroke="#fff" stroke-width="1.6"/>')
        if m in P['lab']:
            lx_, ly_ = P['lab'][m]; w = 140; an = 'l' if lx_ >= X else 'r'
            if ((lx_ - X) ** 2 + (ly_ - Y) ** 2) ** .5 > 16:
                f.svg.append(f'<line x1="{X}" y1="{Y}" x2="{lx_}" y2="{ly_}" stroke="{c}" stroke-opacity=".5" stroke-width="1"/>')
            f.txt(lx_ + 3 if an == 'l' else lx_ - 3 - w, ly_ - 9, w, m, 'lab', f'color:{c};font-weight:{800 if prop else 600};text-align:{"right" if an == "r" else "left"}')
    rx, ry = P['lab']['_guid']; f.txt(rx, ry - 9, 160, 'Guidance methods', 't', f'font-size:12.5px;color:{VIO};font-weight:600;white-space:nowrap')
    r, p = spearmanr(M.rho, M[P['col']])
    f.txt(x1 - 200, y0 + 4 if P['L'] == 'b' else y1 - 30, 196, f'Spearman <i>r</i> = {r:.2f}, <i>p</i> {"&lt; 0.001" if p < .001 else "= %.2f" % p}', 't', f'font-size:12.5px;text-align:right;color:{INK};font-weight:600')
# legend
lx = 8
for g, c in GC.items():
    f.els.append(f'<div class="lg" style="left:{lx}px;top:{H - 30}px"><i style="background:{c}"></i>{g}</div>'); lx += 22 + 7.1 * len(g) + 14
f.els.append(f'<div class="lg" style="left:{lx}px;top:{H - 30}px"><span style="color:{TEALD}">- - -</span> margin sweep</div>')
f.save('Figure5_v1', '.lab{font-size:12.5px;white-space:nowrap;text-shadow:0 0 3px #fbfcfe,0 0 3px #fbfcfe,0 0 3px #fbfcfe,0 0 2px #fbfcfe,0 0 2px #fbfcfe} .lg{font-size:12.5px;white-space:nowrap;width:auto!important} .lg i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}')
