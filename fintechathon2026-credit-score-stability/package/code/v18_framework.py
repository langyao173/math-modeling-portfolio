"""One scientific scene; independent native objects in SVG, PDF and draw.io."""
from pathlib import Path
from figure_io import save_figure
import json, xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle,Ellipse,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
FONT=ROOT/'assets/fonts/ReportChartCJK.ttf'
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
INK='#33404a';RISK='#536548';REF='#8B9C74';CLAY='#C08A78';GRAY='#BBC3C8';PALE='#F4F6F7';GREEN='#F2F5F0'
W,H=1000,448;ITEMS=[]
def box(x,y,w,h,fill='white',stroke=GRAY,dashed=False):ITEMS.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,dashed=dashed))
def txt(x,y,t,size=22,color=INK,bold=False,anchor='middle'):ITEMS.append(dict(kind='text',x=x,y=y,text=t,size=size,color=color,bold=bold,anchor=anchor))
def line(x,y,xx,yy,color=GRAY,width=1.7,arrow=False,dashed=False):ITEMS.append(dict(kind='line',x=x,y=y,x2=xx,y2=yy,color=color,width=width,arrow=arrow,dashed=dashed))
def dot(x,y,r=5,fill=REF):ITEMS.append(dict(kind='ellipse',x=x-r,y=y-r,w=r*2,h=r*2,fill=fill,stroke=fill))

def scene():
    txt(15,24,'一套评分，贯通风险识别、跨期比较与客户复核',24,bold=True,anchor='start')
    box(15,78,150,244,'#F5F6F2','#CDD4C5');txt(90,104,'账户历史',23,bold=True)
    for i,v in enumerate([35,45,39,58,55,73]):box(36+i*19,215-v,10,v,'#A9B5A1','none')
    for y,label in [(247,'余额 · 支用'),(279,'还款 · 应收')]:txt(90,y,label,20)
    txt(90,342,'实际额度 / 敞口',18,color='#75846C')
    line(165,165,202,165,RISK,2,True)
    box(202,78,185,244,'#F5F7F1','#9BAE91');txt(294,103,'固定参考周期',22,color=RISK,bold=True)
    for x in [230,274,318]:box(x,147,26,41,'white','#9BAE91');line(x+13,139,x+13,147,REF)
    line(226,208,360,208,REF,1.4);txt(294,230,'统一重算特征',21,color=RISK)
    txt(294,267,'244 / 414 维',21);txt(294,295,'保留最新账务',19,color='#74816B')
    line(387,165,424,165,RISK,2,True)
    box(424,78,183,244,'#F5F6F2',RISK);txt(515,103,'参考风险预测',22,color=RISK,bold=True)
    for center in [470,558]:
        for a,b in [((center,146),(center-20,169)),((center,146),(center+20,169)),((center-20,169),(center-20,190))]:line(*a,*b,RISK,1.6)
        for x,y in [(center,146),(center-20,169),(center+20,169),(center-20,190)]:dot(x,y,3,RISK)
    txt(515,230,'异构模型融合',20);txt(515,267,'独立概率校准',20);txt(515,295,'p(t)',21,color=RISK)
    line(607,165,644,165,RISK,2,True)
    box(644,78,180,244,'#F3F6F1',RISK);txt(734,103,'稳定与名额协同',22,color=RISK,bold=True)
    # A bounded-update motif; schematic and intentionally unticked.
    box(661,162,144,46,'#DFE9DE','none');line(661,185,805,185,'#A4B89E',1.3,False,True)
    for a,b in [((665,193),(698,168)),((698,168),(733,199)),((733,199),(768,170)),((768,170),(801,170))]:line(*a,*b,RISK,2.7)
    line(768,170,800,131,CLAY,1.8,False,True);dot(801,170,4,RISK)
    txt(734,239,'近稳候选 ±1 pp',19);txt(734,267,'保护 Top-k 名单',19);txt(734,295,'q(t)',21,color=RISK,bold=True)
    line(824,165,861,165,RISK,2,True)
    box(861,78,124,244,'white','#97A589');txt(923,103,'同一输出',22,color=RISK,bold=True)
    for i,v in enumerate([57,46,35,24]):box(880+23*i,200-v,12,v,RISK if i<2 else '#D5DECF','none')
    txt(923,237,'名单排序',21);txt(923,269,'跨期比较',21);txt(923,298,'案件接续',21)
    # Actual exposure bypasses the reference transformation and remains visible.
    line(90,358,90,388,RISK,1.4);line(90,388,923,388,RISK,1.4);line(923,388,923,322,RISK,1.4,True)
    box(238,369,532,35,'white','none');txt(504,387,'实际账务 + 动作凭证 → 证据工具 → 复核 Agent',21,color=RISK)
    txt(500,428,'核实经营动作后沿用参考周期；资质重审、账户变更或模型换版启动重建',18,color='#75846C')

def render():
    scene();width_inches=156/25.4;scale=width_inches*72/W
    f,ax=plt.subplots(figsize=(width_inches,width_inches*H/W));f.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
    svg=ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width='156mm',height=f'{156*H/W}mm',viewBox=f'0 0 {W} {H}')
    ET.SubElement(svg,'title').text='统一评分与智能风险复核'
    defs=ET.SubElement(svg,'defs')
    for color in [INK,RISK,REF,CLAY,GRAY]:
        m=ET.SubElement(defs,'marker',id='a'+color[1:],viewBox='0 0 10 10',refX='9',refY='5',markerWidth='6',markerHeight='6',orient='auto')
        ET.SubElement(m,'path',d='M 0 0 L 10 5 L 0 10 z',fill=color)
    mx=ET.Element('mxfile',host='app.diagrams.net');dia=ET.SubElement(mx,'diagram',id='v17_framework',name='统一评分与客户复核')
    g=ET.SubElement(dia,'mxGraphModel',page='1',pageWidth=str(W),pageHeight=str(H));root=ET.SubElement(g,'root')
    ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    for i,o in enumerate(ITEMS):
        kind=o['kind'];x,y=o['x'],o['y'];oid=str(i+2)
        if kind=='line':
            a={'x1':str(x),'y1':str(y),'x2':str(o['x2']),'y2':str(o['y2']),'stroke':o['color'],'stroke-width':str(o['width'])}
            if o['dashed']:a['stroke-dasharray']='5 4'
            if o['arrow']:a['marker-end']='url(#a'+o['color'][1:]+')'
            ET.SubElement(svg,'line',a)
            ax.add_patch(FancyArrowPatch((x,y),(o['x2'],o['y2']),arrowstyle='-|>' if o['arrow'] else '-',shrinkA=0,shrinkB=0,mutation_scale=6,lw=o['width']*scale,color=o['color'],ls='--' if o['dashed'] else '-'))
            c=ET.SubElement(root,'mxCell',id=oid,parent='1',edge='1',style=f'edgeStyle=none;endArrow={"block" if o["arrow"] else "none"};strokeColor={o["color"]};strokeWidth={o["width"]};dashed={int(o["dashed"])};')
            geo=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
            for label,xx,yy in [('sourcePoint',x,y),('targetPoint',o['x2'],o['y2'])]:ET.SubElement(geo,'mxPoint',x=str(xx),y=str(yy),**{'as':label})
            continue
        if kind=='text':
            sz=o['size'];anchor=o['anchor'];w=max(45,len(o['text'])*sz);h=sz*1.5;xx=x-({'middle':.5,'start':0,'end':1}[anchor])*w;yy=y-h/2
            t=ET.SubElement(svg,'text',x=str(x),y=str(y+sz*.34),fill=o['color'],**{'font-size':str(sz),'font-family':'Droid Sans Fallback, Noto Sans CJK SC, sans-serif','font-weight':'bold' if o['bold'] else 'normal','text-anchor':anchor});t.text=o['text']
            align={'middle':'center','start':'left','end':'right'}[anchor]
            ax.text(x,y,o['text'],ha=align,va='center',fontsize=sz*scale,color=o['color'],fontweight='bold' if o['bold'] else 'normal')
            style=f'html=0;fillColor=none;strokeColor=none;fontSize={sz};fontColor={o["color"]};fontFamily=Droid Sans Fallback;align={align};fontStyle={int(o["bold"])};whiteSpace=wrap;';value=o['text']
        else:
            xx,yy,w,h=x,y,o['w'],o['h'];value='';edge=None if o['stroke']=='none' else o['stroke']
            if kind=='rect':
                ET.SubElement(svg,'rect',x=str(x),y=str(y),width=str(w),height=str(h),fill=o['fill'],stroke=o['stroke'])
                ax.add_patch(Rectangle((x,y),w,h,facecolor=o['fill'],edgecolor=edge,lw=.5))
            else:
                ET.SubElement(svg,'ellipse',cx=str(x+w/2),cy=str(y+h/2),rx=str(w/2),ry=str(h/2),fill=o['fill'],stroke=o['stroke'])
                ax.add_patch(Ellipse((x+w/2,y+h/2),w,h,facecolor=o['fill'],edgecolor=edge,lw=.5))
            style=f'shape={"ellipse" if kind=="ellipse" else "rectangle"};fillColor={o["fill"]};strokeColor={o["stroke"]};'
        c=ET.SubElement(root,'mxCell',id=oid,parent='1',vertex='1',value=value,style=style)
        ET.SubElement(c,'mxGeometry',x=str(xx),y=str(yy),width=str(w),height=str(h),**{'as':'geometry'})
    f.canvas.draw();renderer=f.canvas.get_renderer()
    for t in ax.texts:
        ext=ax.transData.inverted().transform(t.get_window_extent(renderer).get_points());lo,hi=ext.min(0),ext.max(0)
        assert lo[0]>=0 and hi[0]<=W and lo[1]>=0 and hi[1]<=H,(t.get_text(),lo,hi)
    for ext in ['pdf','png']:save_figure(f,OUT/f'v18_framework.{ext}',dpi=350,facecolor='white')
    plt.close(f)
    for ext,obj in [('svg',svg),('drawio',mx)]:ET.ElementTree(obj).write(OUT/f'v18_framework.{ext}',encoding='utf-8',xml_declaration=True)
    (OUT/'v18_framework_scene.json').write_text(json.dumps(ITEMS,ensure_ascii=False,indent=2))
    print(f'{len(ITEMS)} native objects; 156 × {156*H/W:.2f} mm; smallest text {21*scale:.2f} pt')

if __name__=='__main__':render()
