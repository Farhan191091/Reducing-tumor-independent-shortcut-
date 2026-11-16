from common import *
import pandas as pd
R = pd.read_csv('/home/claude/results/out/reliance.csv').merge(pd.read_csv('/home/claude/results/out/abstention.csv'), on='model').set_index('model')
GROUPS = [('Standard', '#475569', ['Baseline', 'ConvNeXt-T (standard head)', 'ResNet-50', 'EfficientNet-B0', 'DeiT-S']),
          ('Guidance', VIO, ['MTL-Seg', 'RRR', 'CAM-Guide', 'CTR-Net']),
          ('Input restriction', TEALD, ['ROI-0.2', 'ROI-0.6', 'ROI-1.5', 'ROI-Mask', 'ROI-Loc']),
          ('Transplantation', '#c98a00', ['TT']),
          ('Both principles', '#e11d48', ['ROI-0.6+TT']),
          ('Skull control', '#94a3b8', ['Baseline-Strip'])]
LABEL = {'ConvNeXt-T (standard head)': 'ConvNeXt-T, std. head'}
RH = 25; GAP = 10; TOP = 70; LW = 290
rows = []; y = TOP
for g, col, ms in GROUPS:
    rows.append(('g', g, col, y));
    for m in ms:
        rows.append(('m', m, col, y)); y += RH
    y += GAP
H = y + 84; W = 1000; f = Fig(W, H)
PANELS = [('a', 'Original slices', 'Balanced accuracy (%)', 'ba_orig', 84, 100, [85, 90, 95, 100], None),
          ('b', 'Tumor reliance <i>ρ</i>', '<i>ρ</i> (95% CI)', 'rho', 0, 1, [0, .25, .5, .75, 1], None),
          ('c', 'Head geometry only', 'Balanced accuracy (%)', 'ba_head_only', 25, 62, [30, 40, 50, 60], 100 / 3)]
PX = [LW + 22, LW + 240, LW + 534]; PWID = [192, 272, 176]
f.box(0, TOP - 26, W, H - TOP + 26 - 4, 'card', 'background:#fbfcfe')
# row stripes, group labels, model names
for kind, name, col, yy in rows:
    if kind == 'g':
        gy0 = yy; n = next(len(ms) for g, c, ms in GROUPS if g == name)
        f.box(10, gy0 + 2, 5, n * RH - 4, '', f'background:{col};border-radius:3px')
        f.txt(22, gy0 + (n * RH) / 2 - 9, 150, name, 't', f'font-size:12.5px;font-weight:700;color:{col};white-space:nowrap')
    else:
        prop = name == 'ROI-0.6+TT'
        if prop:
            f.box(170, yy, W - 180, RH, '', 'background:#fff1f3;border-radius:6px;border:1px solid #fecdd3')
        f.txt(178, yy + 4, 180, LABEL.get(name, name), 't', f'font-size:13px;white-space:nowrap;{"font-weight:800;color:#be123c" if prop else ""}')
yb = rows[-1][3] + RH + 6
for (L, title, ax, col, lo, hi, ticks, chance), px, pw in zip(PANELS, PX, PWID):
    sx = lambda v: px + 10 + (v - lo) / (hi - lo) * (pw - 20)
    f.txt(px - 4, 6, 30, L, 'letter'); f.txt(px + 20, 10, pw, title, 'ptitle', 'font-size:15.5px')
    f.txt(px, yb + 24, pw, ax, 'cap', f'color:{MUT};font-size:12.5px')
    for t in ticks:
        f.svg.append(f'<line x1="{sx(t)}" y1="{TOP - 6}" x2="{sx(t)}" y2="{yb}" stroke="#e5e9f0" stroke-width="1"/>')
        f.txt(sx(t) - 25, yb + 4, 50, f'{t:g}', 'cap', f'font-size:12px;color:{MUT}')
    if chance:
        f.svg.append(f'<line x1="{sx(chance)}" y1="{TOP - 6}" x2="{sx(chance)}" y2="{yb}" stroke="#64748b" stroke-width="1.4" stroke-dasharray="5 4"/>')
        f.txt(sx(chance) + 4, TOP - 22, 60, 'chance', 't', f'font-size:12px;color:{MUT};white-space:nowrap')
    if col == 'rho':
        b = R.loc['Baseline', 'rho']
        f.svg.append(f'<line x1="{sx(b)}" y1="{TOP - 6}" x2="{sx(b)}" y2="{yb}" stroke="#475569" stroke-width="1.2" stroke-dasharray="2 3"/>'); f.txt(sx(b) + 4, TOP - 22, 80, 'Baseline', 't', f'font-size:12px;color:{MUT};white-space:nowrap')
    for kind, name, c, yy in rows:
        if kind != 'm': continue
        v = R.loc[name, col]; cy = yy + RH / 2; prop = name == 'ROI-0.6+TT'
        if col == 'rho':
            lo_, hi_ = R.loc[name, 'rho_lo'], R.loc[name, 'rho_hi']
            f.svg.append(f'<line x1="{sx(0)}" y1="{cy}" x2="{sx(v)}" y2="{cy}" stroke="{c}" stroke-opacity=".25" stroke-width="7" stroke-linecap="round"/>')
            f.svg.append(f'<line x1="{sx(lo_)}" y1="{cy}" x2="{sx(hi_)}" y2="{cy}" stroke="{c}" stroke-width="2"/><line x1="{sx(lo_)}" y1="{cy-5}" x2="{sx(lo_)}" y2="{cy+5}" stroke="{c}" stroke-width="2"/><line x1="{sx(hi_)}" y1="{cy-5}" x2="{sx(hi_)}" y2="{cy+5}" stroke="{c}" stroke-width="2"/>')
            f.txt(sx(hi_) + 9, yy + 4, 40, f'{v:.2f}', 't', f'font-size:12.5px;font-weight:{800 if prop else 600};color:{c}')
        abst = R.loc[name, 'abst_hidden'] > 0
        if col == 'rho' and abst:
            va = R.loc[name, 'rho_answered']; d = 5.5
            f.svg.append(f'<path d="M{sx(va)},{cy-d} L{sx(va)+d},{cy} L{sx(va)},{cy+d} L{sx(va)-d},{cy} Z" fill="#fff" stroke="{c}" stroke-width="1.8"/>')
        if col == 'ba_head_only' and abst:
            f.svg.append(f'<circle cx="{sx(v)}" cy="{cy}" r="{6 if prop else 5}" fill="#fff" stroke="{c}" stroke-width="1.8"/>')
        else:
            f.svg.append(f'<circle cx="{sx(v)}" cy="{cy}" r="{6 if prop else 5}" fill="{c}" stroke="#fff" stroke-width="1.5"/>')
f.els.append(f'<div style="position:absolute;left:{PX[1]}px;top:{H-22}px;font-size:12px;color:{MUT};white-space:nowrap">● <i>ρ</i> (all slices) &nbsp; ◇ <i>ρ</i><sub>ans</sub> (answered slices) &nbsp;&nbsp;&nbsp; ○ abstains (head only)</div>')
f.save('Figure4_v1', '')
