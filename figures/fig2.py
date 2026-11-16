from common import *
W,H=1000,560; f=Fig(W,H)
f.txt(4,2,30,'a','letter'); f.txt(30,6,600,'Four versions of every slice','ptitle')
BY=36; BW=648; f.box(0,BY,BW,H-BY,'band')
S=136; gx=20; x0=66; y0=BY+74
cols=[('Original','expert outline','#e2e8f0',CY,'orig'),('Tumor hidden','test: filled with tissue<br>from the same slice',ROSE,ROSE,'swap'),
      ('Random spot','control: equal area<br>removed elsewhere','#cbd5e1','#cbd5e1','rand'),('Head geometry','brain interior and<br>tumor removed',AMB,AMB,'skull')]
for j,(t,sub,c,_,k) in enumerate(cols):
    x=x0+j*(S+10)
    f.txt(x-6,BY+14,S+12,t,'t',f'color:{c};font-weight:700;font-size:14px;text-align:center;white-space:nowrap')
    f.txt(x-6,BY+34,S+12,sub,'capd','font-size:13px;line-height:1.2')
    for i,cl in enumerate('gmp'):
        f.im(f'h_{cl}_{k}.png',x,y0+i*(S+8),S)
for i,cl in enumerate(['glioma','meningioma','pituitary']):
    yc=y0+i*(S+8)+S/2
    f.els.append(f'<div class="rowl" style="left:{-40}px;top:{yc-11}px;width:150px;color:{COLL[cl]}">{cl.capitalize()}</div>')
# panel b
PX=BW+14; PW=W-PX
f.box(PX,BY,PW,H-BY,'card','background:linear-gradient(170deg,#fff5f6 0%,#ffffff 45%);border-color:#f3d2d7')
f.txt(PX+2,2,30,'b','letter'); f.txt(PX+28,6,300,'Tumor reliance score <i>ρ</i>','ptitle')
f.box(PX+16,BY+16,PW-32,58,'formula','font-size:18px','<span><i>ρ</i> = <span class="frac"><span>BA<sub>orig</sub> − BA<sub>hid</sub></span><span>BA<sub>orig</sub> − 1/<i>K</i></span></span></span>')
# schematic bars
bx=PX+34; base=BY+262; top=BY+146; hh=base-top
yv=lambda p: base-hh*p
f.svg.append(f'<line x1="{bx-8}" y1="{base}" x2="{PX+PW-22}" y2="{base}" stroke="#334155" stroke-width="1.2"/>')
f.svg.append(f'<line x1="{bx-8}" y1="{yv(1/3)}" x2="{PX+PW-22}" y2="{yv(1/3)}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="5 4"/>')
f.txt(PX+PW-90,yv(1/3)-19,70,'chance','t',f'color:{MUT};text-align:right')
for gx_,lab,o,h in [(bx+8,'ignores<br>the tumor',.95,.88),(bx+158,'needs<br>the tumor',.93,.36)]:
    f.svg.append(f'<rect x="{gx_}" y="{yv(o)}" width="34" height="{base-yv(o)}" rx="5" fill="#334155"/>')
    f.svg.append(f'<rect x="{gx_+42}" y="{yv(h)}" width="34" height="{base-yv(h)}" rx="5" fill="{ROSE}"/>')
    r=(o-h)/(o-1/3)
    f.txt(gx_-22,yv(o)-27,120,f'<i>ρ</i> ≈ {r:.1f}','t',f'text-align:center;font-weight:800;font-size:16px;color:{"#be123c" if r>.5 else MUT}')
    f.txt(gx_-22,base+7,120,lab,'cap')
f.els.append(f'<div class="lg" style="left:{bx-6}px;top:{BY+86}px"><i style="background:#334155"></i>original<i style="background:{ROSE};margin-left:14px"></i>tumor hidden<span style="color:{MUT};margin-left:10px;font-style:italic">schematic</span></div>')
# reading guide
ry=BY+312
for i,(c,t1,t2) in enumerate([(ROSE,'Tumor hidden','accuracy should fall toward chance when the model needs the tumor'),
                              ('#94a3b8','Random spot hidden','accuracy should not change (validity control)'),
                              ('#d99a06','Head geometry only','accuracy above chance exposes a shortcut')]):
    yy=ry+i*66
    f.box(PX+16,yy+3,5,52,'','background:%s;border-radius:3px'%c)
    f.txt(PX+30,yy,PW-46,f'<b>{t1}</b><br>{t2}','t','line-height:1.38')
f.save('Figure2_v3',f'''.rowl{{font-size:14px;font-weight:700;text-align:center;transform:rotate(-90deg);white-space:nowrap}}
.frac{{display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;margin-left:6px}} .frac span:first-child{{border-bottom:1.2px solid #2a1a66;padding:0 6px 2px}} .frac span:last-child{{padding-top:2px}}
.lg{{font-size:13px;white-space:nowrap;width:auto!important}} .lg i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}}''')
