from common import *
import pandas as pd
S = pd.read_csv('/home/claude/results/out/safety.csv').set_index('model')
T = '/home/claude/figdata/figure_assets/tables/'
INV = {'CNX-Plain': 'Baseline', 'ResNet-50': 'ResNet-50', 'EfficientNet-B0': 'EfficientNet-B0', 'ConvNeXt-Tiny': 'ConvNeXt-T (standard head)', 'DeiT-Small': 'DeiT-S',
       'MTL-Seg': 'MTL-Seg', 'RRR': 'RRR', 'CAM-Guide': 'CAM-Guide', 'CTR-Net': 'CTR-Net', 'ROI-Crop': 'ROI-0.2', 'ROI-Crop-m06': 'ROI-0.6', 'ROI-Crop-m15': 'ROI-1.5',
       'ROI-Mask': 'ROI-Mask', 'ROI-Loc': 'ROI-Loc', 'CNX-Transplant': 'TT', 'ROI-m06-Transplant': 'ROI-0.6+TT', 'CNX-Strip': 'Baseline-Strip'}
cf = pd.read_csv(T + 'table_conformal.csv'); cf = cf[(cf.alpha == 0.05) & (cf.Set == 'BraTS')]; COV = {INV[m]: v for m, v in zip(cf.Model, cf['Coverage (%)']) if m in INV}
al = pd.read_csv(T + 'table_shortcut_alarm.csv'); al = al[al.Set.str.contains('tumour-free')]
ALARM = {INV[m]: (c, a) for m, c, a in zip(al.Model, al['Tumour-free confident >= 0.85 (%)'], al['Tumour-free answered by alarm rule (%)'])}
GROUPS = [('Standard', '#475569', ['Baseline', 'ConvNeXt-T (standard head)', 'ResNet-50', 'EfficientNet-B0', 'DeiT-S']),
          ('Guidance', VIO, ['MTL-Seg', 'RRR', 'CAM-Guide', 'CTR-Net']),
          ('Input restriction', TEALD, ['ROI-0.2', 'ROI-0.6', 'ROI-1.5', 'ROI-Mask', 'ROI-Loc']),
          ('Transplantation', '#c98a00', ['TT']), ('Both principles', '#e11d48', ['ROI-0.6+TT']), ('Skull control', '#94a3b8', ['Baseline-Strip'])]
LABEL = {'ConvNeXt-T (standard head)': 'ConvNeXt-T, std.'}
RH = 25; GAP = 10; TOP = 92; LW = 290
rows = []; y = TOP
for g, col, ms in GROUPS:
    rows.append(('g', g, col, y))
    for m in ms: rows.append(('m', m, col, y)); y += RH
    y += GAP
H = y + 70; W = 1000; f = Fig(W, H)
f.box(0, TOP - 48, W, H - TOP + 48 - 4, 'card', 'background:#fbfcfe')
for kind, name, col, yy in rows:
    if kind == 'g':
        n = next(len(ms) for g, c, ms in GROUPS if g == name)
        f.box(10, yy + 2, 5, n * RH - 4, '', f'background:{col};border-radius:3px')
        f.txt(22, yy + (n * RH) / 2 - 9, 150, name, 't', f'font-size:12.5px;font-weight:700;color:{col};white-space:nowrap')
    else:
        prop = name == 'ROI-0.6+TT'
        if prop: f.box(170, yy, W - 180, RH, '', 'background:#fff1f3;border-radius:6px;border:1px solid #fecdd3')
        f.txt(178, yy + 4, 180, LABEL.get(name, name), 't', f'font-size:13px;white-space:nowrap;{"font-weight:800;color:#be123c" if prop else ""}')
yb = rows[-1][3] + RH + 6
# ---- a: what the models answer on tumor-free slices ----
ax0, aw = LW + 28, 262; sx = lambda v: ax0 + v / 100 * aw
f.txt(ax0 - 26, 6, 30, 'a', 'letter'); f.txt(ax0, 10, 340, 'Answers on 911 tumor-free slices', 'ptitle', 'font-size:15.5px')
CLS = [('free_glioma', COL['glioma'], 'glioma'), ('free_meningioma', COL['meningioma'], 'meningioma'), ('free_pituitary', COL['pituitary'], 'pituitary'), ('free_abstain', '#cbd5e1', 'abstained')]
lx = ax0
for _, c, t in CLS:
    f.els.append(f'<div class="lg" style="left:{lx}px;top:{TOP - 40}px"><i style="background:{c}"></i>{t}</div>'); lx += 16 + 6.6 * len(t) + 10
f.els.append(f'<div class="lg" style="left:{ax0}px;top:{TOP - 22}px"><b style="display:inline-block;width:3px;height:13px;background:#0f172a;margin-right:6px;vertical-align:-2px"></b>confident (max. probability ≥ 0.85)</div>')
for t in [0, 25, 50, 75, 100]:
    f.svg.append(f'<line x1="{sx(t)}" y1="{TOP - 2}" x2="{sx(t)}" y2="{yb}" stroke="#e5e9f0"/>'); f.txt(sx(t) - 25, yb + 4, 50, f'{t}', 'cap', f'font-size:12px;color:{MUT}')
f.txt(ax0, yb + 22, aw, 'Share of tumor-free slices (%)', 'cap', f'color:{MUT};font-size:12.5px')
for kind, name, c, yy in rows:
    if kind != 'm': continue
    x = 0
    for k, cc, _ in CLS:
        v = S.loc[name, k]
        if v > 0: f.svg.append(f'<rect x="{sx(x)}" y="{yy + 5}" width="{aw * v / 100}" height="{RH - 10}" fill="{cc}" fill-opacity=".85"/>')
        x += v
    cv = S.loc[name, 'free_confident']
    f.svg.append(f'<rect x="{sx(cv) - 1.5}" y="{yy + 1}" width="3" height="{RH - 2}" fill="#0f172a"/>')
    f.txt(sx(100) + 6, yy + 4, 50, f'{cv:.1f}', 't', f'font-size:12.5px;font-weight:{800 if name == "ROI-0.6+TT" else 600};color:{c}')
# ---- b: conformal coverage on BraTS ----
bx0, bw = LW + 370, 132; lo, hi = 75, 100; sx2 = lambda v: bx0 + (v - lo) / (hi - lo) * bw
f.txt(bx0 - 26, 6, 30, 'b', 'letter'); f.txt(bx0, 10, 190, 'Coverage', 'ptitle', 'font-size:15.5px')
f.txt(bx0 - 10, TOP - 40, bw + 20, 'conformal, BraTS', 'cap', f'color:{MUT};font-size:12.5px')
for t in [75, 80, 85, 90, 95, 100]:
    f.svg.append(f'<line x1="{sx2(t)}" y1="{TOP - 2}" x2="{sx2(t)}" y2="{yb}" stroke="{"#e11d48" if t == 95 else "#e5e9f0"}" {"stroke-dasharray=\"5 4\" stroke-width=\"1.4\"" if t == 95 else ""}/>')
    if t in (80, 90, 100): f.txt(sx2(t) - 20, yb + 4, 40, f'{t}', 'cap', f'font-size:12px;color:{MUT}')
f.txt(sx2(95) - 30, TOP - 22, 60, 'target', 't', f'font-size:12px;color:#e11d48;text-align:center'); f.txt(bx0, yb + 22, bw, 'Coverage (%)', 'cap', f'color:{MUT};font-size:12.5px')
for kind, name, c, yy in rows:
    if kind != 'm' or name not in COV: continue
    v = COV[name]; cy = yy + RH / 2
    f.svg.append(f'<line x1="{sx2(95)}" y1="{cy}" x2="{sx2(v)}" y2="{cy}" stroke="{c}" stroke-opacity=".35" stroke-width="3"/>')
    f.svg.append(f'<circle cx="{sx2(v)}" cy="{cy}" r="{6 if name == "ROI-0.6+TT" else 5}" fill="{c}" stroke="#fff" stroke-width="1.5"/>')
# ---- c: shortcut alarm (full-slice models only) ----
cx0, cw = LW + 550, 140; sx3 = lambda v: cx0 + v / 60 * cw
f.txt(cx0 - 26, 6, 30, 'c', 'letter'); f.txt(cx0, 10, 160, 'Shortcut alarm', 'ptitle', 'font-size:15.5px')
f.txt(cx0 - 14, TOP - 40, cw + 28, 'tumor-free, answered (%)', 'cap', f'color:{MUT};font-size:12.5px')
for t in [0, 20, 40, 60]:
    f.svg.append(f'<line x1="{sx3(t)}" y1="{TOP - 2}" x2="{sx3(t)}" y2="{yb}" stroke="#e5e9f0"/>'); f.txt(sx3(t) - 20, yb + 4, 40, f'{t}', 'cap', f'font-size:12px;color:{MUT}')
f.txt(cx0 - 10, yb + 22, cw + 20, 'Share (%)', 'cap', f'color:{MUT};font-size:12.5px')
for kind, name, c, yy in rows:
    if kind != 'm': continue
    cy = yy + RH / 2
    if name in ALARM:
        cv, av = ALARM[name]
        f.svg.append(f'<circle cx="{sx3(cv)}" cy="{cy}" r="4.5" fill="none" stroke="{c}" stroke-width="1.6"/>')
        f.svg.append(f'<path d="M{sx3(cv) - 6},{cy} L{sx3(av) + 9},{cy}" stroke="{c}" stroke-width="1.6" fill="none" marker-end="url(#ahk)"/>')
        f.svg.append(f'<circle cx="{sx3(av)}" cy="{cy}" r="4.5" fill="{c}"/>')
    elif kind == 'm' and S.loc[name, 'free_abstain'] > 0:
        f.txt(cx0, yy + 5, cw, 'abstains instead', 't', f'font-size:11.5px;color:{MUT};font-style:italic;white-space:nowrap')
f.els.append(f'<div class="lg" style="left:{cx0 - 10}px;top:{TOP - 22}px"><i style="background:none;border:1.6px solid #475569"></i>p ≥ 0.85 <i style="background:#475569;margin-left:8px"></i>alarm</div>')
f.save('Figure6_v1', '.lg{font-size:12px;white-space:nowrap;width:auto!important} .lg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}')
