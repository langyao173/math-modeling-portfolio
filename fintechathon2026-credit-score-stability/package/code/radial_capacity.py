"""Capacity comparison: a radial evidence chart from frozen row-level evaluations.

One scene drives the paper PDF, SVG and separately editable draw.io objects.
Inner bar LENGTH is linear in observed events, on the same 0--99 scale.
The independent outer annulus plots observed events / selected customers.
"""
from pathlib import Path
from figure_io import save_figure
import base64
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
W, H = 1200, 618
WIDTH_IN = 6.25
INK, MUTED, GRID = '#293B39', '#6D7774', '#D9DEDA'
GREEN = '#416D58'
METHODS = [
    ('A', 'old_actual179_ensemble', '动作专属集成', '#8B9AA8'),
    ('B', 'pool179_ensemble', '历史池 / 179维', '#A595AD'),
    ('C', 'cashflow_ensemble', '现金流 / 244维', '#C9A17A'),
    ('D', 'target_weighted_ensemble', '目标加权 / 244维', GREEN),
]
CAPACITIES = [.05, .1, .2, .4]
SCENE = []


def text(x, y, value, size=24, color=INK, bold=False, align='center', rotation=0):
    SCENE.append(dict(kind='text', x=x, y=y, value=str(value), size=size,
                      color=color, bold=bold, align=align, rotation=rotation))


def path(points, fill='none', stroke=GRID, width=1, closed=False):
    SCENE.append(dict(kind='path', points=points, fill=fill, stroke=stroke,
                      width=width, closed=closed))


def rect(x, y, w, h, fill, stroke='none', width=1):
    SCENE.append(dict(kind='rect', x=x, y=y, w=w, h=h, fill=fill,
                      stroke=stroke, width=width))


def circle(x, y, r, fill='none', stroke=GRID, width=1):
    SCENE.append(dict(kind='circle', x=x, y=y, r=r, fill=fill,
                      stroke=stroke, width=width))


CX, CY = 294, 303
R0, R1 = 49, 171
P0, P1 = 190, 237


def polar(angle, radius):
    t = math.radians(angle)
    return CX + math.sin(t)*radius, CY - math.cos(t)*radius


def arc(a, b, radius, step=1.2):
    n = max(2, math.ceil(abs(b-a)/step))
    return [polar(a+(b-a)*i/n, radius) for i in range(n+1)]


def sector(a, b, inner, outer, fill, stroke='none', width=1):
    path(arc(a, b, outer)+list(reversed(arc(a, b, inner))), fill, stroke, width, True)


def radial_bar(angle, radius, width, fill):
    # A constant-width rectangular bar: its LENGTH, not its area, carries data.
    a, b = polar(angle, R0), polar(angle, radius)
    t = math.radians(angle)
    dx, dy = math.cos(t)*width/2, math.sin(t)*width/2
    points = [(a[0]-dx, a[1]-dy), (b[0]-dx, b[1]-dy),
              (b[0]+dx, b[1]+dy), (a[0]+dx, a[1]+dy)]
    path(points, fill, 'none', 0, True)


def load_records():
    file = ROOT/'results/legacy_evidence/capacity.csv'
    records = []
    keys = {m[1] for m in METHODS}
    for r in csv.DictReader(file.open()):
        if r['group'] != 'action' or r['model'] not in keys:
            continue
        d = {k: r[k] for k in ('group', 'model')}
        d.update({k: int(r[k]) for k in ('n', 'k', 'hits', 'unknown_selected', 'events', 'unknown')})
        d['capacity'] = float(r['capacity'])
        d['observed_yield_percent'] = 100*d['hits']/d['k']
        d['observed_event_coverage_percent'] = 100*d['hits']/d['events']
        records.append(d)
    assert len(records) == 16
    assert {r['events'] for r in records} == {99}
    assert {r['n'] for r in records} == {2907}
    assert len({(r['capacity'], r['model']) for r in records}) == 16
    return records, file


def scene(records):
    SCENE.clear()
    lookup = {(r['capacity'], r['model']): r for r in records}
    text(12, 20, 'a  容量与复核效率', 28, bold=True, align='left')
    text(622, 20, 'b  主容量比较', 28, bold=True, align='left')

    for j, cap in enumerate(CAPACITIES):
        start, end = j*90+5, (j+1)*90-5
        main = cap == .1
        sector(start, end, R0, R1, '#EDF3EF' if main else '#F3F4F6')
        sector(start, end, P0, P1, '#F1F6F2' if main else '#FBFBFC')
        sector(start, end, 244, 265, '#DBE7DF' if main else '#E8EAEF')
        path(arc(start, end, R1), stroke=GREEN if main else '#A9B3B0', width=1.6)
        for value in [25, 50, 75]:
            path(arc(start, end, R0+(R1-R0)*value/99), stroke=GRID, width=.95)
        for value in [0, 10, 20]:
            path(arc(start, end, P0+(P1-P0)*value/20), stroke='#DCE2DE', width=.8)
        dots = []
        for i, (letter, model, label, color) in enumerate(METHODS):
            angle = start+(end-start)*(i+.5)/4
            d = lookup[(cap, model)]
            path([polar(angle, R0), polar(angle, 266)], stroke='#DEE3DF', width=.85)
            radius = R0+(R1-R0)*d['hits']/99
            radial_bar(angle, radius, 10.5, color)
            x, y = polar(angle, radius+13)
            rotation = -angle if angle <= 90 or angle >= 270 else 180-angle
            text(x, y, d['hits'], 23, color, bold=(letter == 'D'), rotation=rotation)
            x, y = polar(angle, 255)
            text(x, y, letter, 23, color, bold=True, rotation=rotation)
            dots.append(polar(angle, P0+(P1-P0)*d['observed_yield_percent']/20))
        path(dots, stroke='#7A8580', width=1.65)
        for point, method in zip(dots, METHODS):
            circle(*point, 3.9, 'white', method[3], 1.8)
        x, y = polar(j*90+45, 291)
        text(x, y, f'{cap:.0%} 容量', 26, GREEN if main else INK,
             bold=main, rotation=-(j*90+45) if j in [0,3] else 180-(j*90+45))

    circle(CX, CY, R0, 'white', '#B6C1BA', 1.2)
    # Numbered radial axes in the two deliberate breaks between sectors.
    for value in [0, 25, 50, 75, 99]:
        x,y=polar(0, R0+(R1-R0)*value/99)
        rect(x-13, y-10, 26, 20, 'white')
        text(x, y, value, 20, MUTED)
    for value in [0, 10, 20]:
        x,y=polar(180, P0+(P1-P0)*value/20)
        rect(x-18, y-10, 36, 20, 'white')
        text(x, y, f'{value}%', 19.5, MUTED)
    text(CX, CY-11, '2,907', 25, bold=True)
    text(CX, CY+16, '同一队列', 21, MUTED)

    # Explicit, compact explanation of the two quantitative encodings.
    rect(627, 61, 10, 28, GREEN)
    text(650, 75, '内圈柱：覆盖事件数（例）', 24, align='left')
    path([(627, 111),(637, 102),(646, 111)], stroke='#7A8580', width=1.9)
    circle(637,102,3,'white',GREEN,1.6)
    text(661, 110, '外圈线：已观察命中率（%）', 24, align='left')
    text(627, 152, '10% 容量 / 291 个复核名额', 27, bold=True, align='left')

    # Same-order four-method horizontal bars make exact main-budget comparison easy.
    BX, BW, MAX = 659, 425, 55
    for i,(letter, model, label, color) in enumerate(METHODS):
        d=lookup[(.1,model)]
        y=200+i*77
        circle(636,y,12,color,'none',0)
        text(636,y,letter,19,'white',True)
        text(BX,y,label,25,color=GREEN if letter=='D' else INK,
             bold=letter=='D',align='left')
        rect(BX,y+17,BW,14,'#F0F2F0')
        rect(BX,y+17,BW*d['hits']/MAX,14,color)
        text(BX+BW*d['hits']/MAX+16,y+24,d['hits'],27,color,
             bold=letter=='D',align='left')
    path([(627,490),(1179,490)],stroke='#D3DBD6',width=1.3)
    text(627,523,'目标加权覆盖 48 / 99 例事件',27,GREEN,True,align='left')
    text(627,558,'事件覆盖率 48.5%    已观察命中率 16.5%',23,MUTED,align='left')
    text(12,601,'统一刻度：内圈 0–99 例；外圈 0–20%。A–D 在各扇区及右侧对应同一模型。',22,MUTED,align='left')


def svg_and_drawio(out):
    svg=ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width='158.75mm',height=f'{158.75*H/W}mm',viewBox=f'0 0 {W} {H}')
    ET.SubElement(svg,'title').text='同一提额队列的容量覆盖与复核效率'
    ET.SubElement(svg,'desc').text='内圈为统一零起点的覆盖事件数，外圈为覆盖事件数除以名单人数。四档容量及四类同序对照，右侧放大10%容量比较。'
    defs=ET.SubElement(svg,'defs')
    # Keep live text while making the standalone SVG portable across machines.
    fonts=[('CapacityCJK',ROOT/'assets/fonts/ReportChartCJK.ttf'),('CapacityLatin',ROOT/'assets/fonts/WorkbenchLatin.ttf')]
    font_rules=[]
    for family,file in fonts:
        if file.exists():
            font_rules.append("@font-face{font-family:'"+family+"';src:url(data:font/ttf;base64,"+base64.b64encode(file.read_bytes()).decode()+") format('truetype');}")
    ET.SubElement(defs,'style').text='\n'.join(font_rules)
    mx=ET.Element('mxfile',host='app.diagrams.net')
    diagram=ET.SubElement(mx,'diagram',name='容量与复核效率',id='capacity-ring')
    graph=ET.SubElement(diagram,'mxGraphModel',page='1',pageWidth=str(W),pageHeight=str(H),grid='0')
    root=ET.SubElement(graph,'root')
    ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    for i,o in enumerate(SCENE):
        k=o['kind'];cellid=f'object-{i+1}'
        if k=='text':
            anchor={'left':'start','center':'middle','right':'end'}[o['align']]
            a={'x':str(o['x']),'y':str(o['y']+o['size']*.33),'fill':o['color'],'font-size':str(o['size']),'font-family':'CapacityLatin, CapacityCJK, sans-serif','font-weight':'bold' if o['bold'] else 'normal','text-anchor':anchor}
            if o['rotation']:a['transform']=f"rotate({-o['rotation']} {o['x']} {o['y']})"
            ET.SubElement(svg,'text',a).text=o['value']
            width=sum(1 if ord(c)>127 else .57 for c in o['value'])*o['size']+12
            height=o['size']*1.45
            x=o['x']-{'left':0,'center':.5,'right':1}[o['align']]*width;y=o['y']-height/2
            style=f"text;html=0;whiteSpace=wrap;overflow=visible;spacing=0;align={o['align']};verticalAlign=middle;fontFamily=Noto Sans CJK SC;fontSize={o['size']};fontColor={o['color']};fontStyle={1 if o['bold'] else 0};rotation={-o['rotation']};"
            val=o['value']
        elif k=='rect':
            x,y,width,height=o['x'],o['y'],o['w'],o['h'];val=''
            ET.SubElement(svg,'rect',x=str(x),y=str(y),width=str(width),height=str(height),fill=o['fill'],stroke=o['stroke'],**{'stroke-width':str(o['width'])})
            style=f"shape=rectangle;fillColor={o['fill']};strokeColor={o['stroke']};strokeWidth={o['width']};"
        elif k=='circle':
            x,y,width,height=o['x']-o['r'],o['y']-o['r'],o['r']*2,o['r']*2;val=''
            ET.SubElement(svg,'circle',cx=str(o['x']),cy=str(o['y']),r=str(o['r']),fill=o['fill'],stroke=o['stroke'],**{'stroke-width':str(o['width'])})
            style=f"shape=ellipse;fillColor={o['fill']};strokeColor={o['stroke']};strokeWidth={o['width']};"
        else:
            pts=o['points'];val=''
            d='M '+' L '.join(f'{a:.4f},{b:.4f}' for a,b in pts)+(' Z' if o['closed'] else '')
            ET.SubElement(svg,'path',d=d,fill=o['fill'],stroke=o['stroke'],**{'stroke-width':str(o['width']),'stroke-linejoin':'round'})
            x=min(p[0] for p in pts);y=min(p[1] for p in pts)
            width=max(max(p[0] for p in pts)-x,.1);height=max(max(p[1] for p in pts)-y,.1)
            sh=ET.Element('shape',w=str(width),h=str(height),aspect='variable',strokewidth='inherit')
            bg=ET.SubElement(sh,'background');q=ET.SubElement(bg,'path')
            for j,(px,py) in enumerate(pts):ET.SubElement(q,'move' if j==0 else 'line',x=str(px-x),y=str(py-y))
            if o['closed']:ET.SubElement(q,'close')
            fg=ET.SubElement(sh,'foreground');ET.SubElement(fg,'fillstroke' if o['closed'] else 'stroke')
            # Inline mxGraph stencils remain independent editable vector shapes.
            compressor=zlib.compressobj(wbits=-15)
            raw=compressor.compress(ET.tostring(sh,encoding='utf-8'))+compressor.flush()
            stencil=base64.b64encode(raw).decode()
            style=f"shape=stencil({stencil});fillColor={o['fill']};strokeColor={o['stroke']};strokeWidth={o['width']};"
        cell=ET.SubElement(root,'mxCell',id=cellid,parent='1',vertex='1',value=val,style=style)
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(width),height=str(height),**{'as':'geometry'})
    ET.ElementTree(svg).write(str(out)+'.svg',encoding='utf-8',xml_declaration=True)
    ET.ElementTree(mx).write(str(out)+'.drawio',encoding='utf-8',xml_declaration=True)


def render(out):
    font=ROOT/'assets/fonts/ReportChartCJK.ttf'
    font_manager.fontManager.addfont(str(font))
    cjk=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.family':['DejaVu Sans',cjk],'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    scale=WIDTH_IN*72/W
    f,ax=plt.subplots(figsize=(WIDTH_IN,WIDTH_IN*H/W))
    f.subplots_adjust(0,0,1,1);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
    for o in SCENE:
        k=o['kind']
        if k=='text':
            ax.text(o['x'],o['y'],o['value'],fontsize=o['size']*scale,
                    color=o['color'],weight='bold' if o['bold'] else 'normal',
                    ha=o['align'],va='center',rotation=o['rotation'],rotation_mode='anchor')
        elif k=='circle':
            ax.add_patch(Circle((o['x'],o['y']),o['r'],facecolor=o['fill'],edgecolor=o['stroke'],lw=o['width']*scale))
        elif k=='rect':
            ax.add_patch(Rectangle((o['x'],o['y']),o['w'],o['h'],facecolor=o['fill'],edgecolor=o['stroke'],lw=o['width']*scale))
        else:
            ax.add_patch(Polygon(o['points'],closed=o['closed'],facecolor=o['fill'],edgecolor=o['stroke'],lw=o['width']*scale,joinstyle='round'))
    f.canvas.draw()
    renderer=f.canvas.get_renderer()
    for t in ax.texts:
        box=t.get_window_extent(renderer).transformed(ax.transData.inverted())
        assert box.xmin>=-1 and box.xmax<=W+1 and box.ymin>=-1 and box.ymax<=H+1,(t.get_text(),box)
    for ext in ['pdf','png']:save_figure(f,str(out)+'.'+ext,dpi=450,facecolor='white')
    plt.close(f)


def main():
    OUT.mkdir(exist_ok=True)
    records,source=load_records();scene(records)
    out=OUT/'v17_primary_capacity'
    render(out);svg_and_drawio(out)
    with (OUT/'v17_primary_capacity_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (OUT/'v17_primary_capacity_scene.json').write_text(json.dumps(SCENE,ensure_ascii=False,indent=2))
    metadata={'source':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'records':len(records),'objects':len(SCENE),'bar_scale':[0,99],'outer_line_scale_percent':[0,20],
              'bar_encoding':'constant width, linear length from an annular zero baseline',
              'outer_metric':'100 * observed events / all selected customers',
              'selection':'same four incumbent comparisons and four capacities as the replaced line chart',
              'source_numbers_changed':False}
    (OUT/'v17_primary_capacity_sources.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':main()
