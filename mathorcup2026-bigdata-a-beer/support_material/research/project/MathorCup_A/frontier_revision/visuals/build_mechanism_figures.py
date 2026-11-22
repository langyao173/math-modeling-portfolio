#!/usr/bin/env python3
"""Build research diagrams as SVG and individually editable draw.io primitives.

The same scene drives both exports. Every plotted geometric shape is schematic;
scientific dependencies are registered in Figure_facts.md. No raster is embedded.

"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zlib

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "project" / "MathorCup_A"
FONT = PROJECT / "assets/fonts/NotoSansCJKsc-Regular.otf"
FAMILY = "Noto Sans CJK SC"
C = dict(ink="#263B43", text="#53656D", teal="#166B68", teal2="#4D9993",
         pale="#E8F1EF", faint="#F3F6F6", orange="#CB8748", ochre="#E7C7A2",
         gray="#B5C0C3", line="#CFD8DA", white="#FFFFFF", reject="#A96A75")


def tint(hex_color, amount):
    """Mix a base color with white; amount is the retained base fraction."""
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(255 + amount*(v-255)):02X}" for v in rgb)


def clip_halfspace(points, a, b, c):
    """Clip a schematic polygon against a*x+b*y <= c."""
    out=[]
    for p,q in zip(points,points[1:]+points[:1]):
        fp=a*p[0]+b*p[1]-c; fq=a*q[0]+b*q[1]-c
        if fp<=0:out.append(p)
        if (fp<=0)!=(fq<=0):
            t=fp/(fp-fq);out.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
    return out


def font(size, bold=False):
    return ImageFont.truetype(str(FONT), round(size))


def measure(s, size):
    f = font(size)
    return max((f.getlength(line) for line in s.split("\n")), default=0)


class Scene:
    def __init__(self, name, title, w, h, caption):
        self.name, self.title, self.w, self.h, self.caption = name, title, w, h, caption
        self.el = []

    def add(self, kind, **a):
        a.update(kind=kind, id=f"{self.name}_{len(self.el)+1:04d}")
        a.setdefault("facts", "schematic")
        self.el.append(a)
        return a["id"]

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.5, radius=0, **kw):
        return self.add("rect", x=x, y=y, w=w, h=h, fill=fill,
                        stroke=stroke or "none", sw=sw, radius=radius, **kw)

    def ellipse(self, x, y, w, h, fill="none", stroke=None, sw=1.5, rotate=0, **kw):
        return self.add("ellipse", x=x, y=y, w=w, h=h, fill=fill,
                        stroke=stroke or "none", sw=sw, rotate=rotate, **kw)

    def line(self, pts, color=None, sw=2, dash=False, arrow=False, **kw):
        return self.add("line", pts=pts, stroke=color or C["ink"], sw=sw,
                        dash=dash, arrow=arrow, **kw)

    def poly(self, pts, fill="none", stroke=None, sw=1.5, **kw):
        return self.add("polygon", pts=pts, fill=fill, stroke=stroke or "none", sw=sw, **kw)

    def text(self, x, y, s, size=25, color=None, bold=False, anchor="middle", **kw):
        # At 16.1 cm paper width, the ordinary 26 px labels remain about 7.4 pt.
        # Small in-motif labels have a 23 px floor; SVG and draw.io share it.
        size=max(23,size+3)
        return self.add("text", x=x, y=y, s=s, size=size, color=color or C["ink"],
                        bold=bold, anchor=anchor, width=measure(s, size)+8,
                        height=size*1.5*len(s.split("\n")), **kw)

    def dot(self, x, y, r=5, fill=None, stroke=None, sw=1.5, **kw):
        return self.ellipse(x-r, y-r, 2*r, 2*r, fill=fill or C["teal"], stroke=stroke, sw=sw, **kw)

    def arrow(self, x1, y1, x2, y2, **kw):
        return self.line([(x1,y1),(x2,y2)],arrow=True,**kw)

    def svg(self, path):
        ns="http://www.w3.org/2000/svg"
        ET.register_namespace("",ns)
        root=ET.Element(f"{{{ns}}}svg", width=str(self.w), height=str(self.h),
                        viewBox=f"0 0 {self.w} {self.h}", role="img")
        ET.SubElement(root,"title").text=self.title
        ET.SubElement(root,"desc").text=self.caption
        defs=ET.SubElement(root,"defs")
        colors={a["stroke"] for a in self.el if a["kind"]=="line" and a["arrow"]}
        for c in sorted(colors):
            m=ET.SubElement(defs,"marker",id="arrow"+c[1:],viewBox="0 0 10 10",
                            refX="9",refY="5",markerWidth="10",markerHeight="10",
                            orient="auto-start-reverse",markerUnits="userSpaceOnUse")
            ET.SubElement(m,"path",d="M 0 1 L 9 5 L 0 9 Z",fill=c)
        ET.SubElement(root,"rect",x="0",y="0",width=str(self.w),height=str(self.h),fill="white")
        for a in self.el:
            common={"id":a["id"],"data-facts":a["facts"]}
            k=a["kind"]
            if k=="text":
                el=ET.SubElement(root,"text",common|dict(x=str(a["x"]),y=str(a["y"]),
                    fill=a["color"], **{"font-family":FAMILY+", sans-serif","font-size":str(a["size"]),
                    "font-weight":"600" if a["bold"] else "400", "text-anchor":a["anchor"],
                    "dominant-baseline":"central"}))
                lines=a["s"].split("\n")
                if len(lines)==1:el.text=a["s"]
                else:
                    for j,t in enumerate(lines):
                        ET.SubElement(el,"tspan",x=str(a["x"]),dy=str(-.65*a["size"]*(len(lines)-1) if j==0 else 1.3*a["size"])).text=t
            elif k=="line":
                d=common|dict(points=" ".join(f"{x:g},{y:g}" for x,y in a["pts"]),
                    fill="none",stroke=a["stroke"],**{"stroke-width":str(a["sw"]),"stroke-linejoin":"round","stroke-linecap":"round"})
                if a["dash"]:d["stroke-dasharray"]="6 6"
                if a["arrow"]:d["marker-end"]="url(#arrow"+a["stroke"][1:]+")"
                ET.SubElement(root,"polyline",d)
            elif k=="polygon":
                ET.SubElement(root,"polygon",common|dict(points=" ".join(f"{x:g},{y:g}" for x,y in a["pts"]),fill=a["fill"],stroke=a["stroke"],**{"stroke-width":str(a["sw"]),"stroke-linejoin":"round"}))
            elif k=="rect":
                ET.SubElement(root,"rect",common|dict(x=str(a["x"]),y=str(a["y"]),width=str(a["w"]),height=str(a["h"]),rx=str(a["radius"]),fill=a["fill"],stroke=a["stroke"],**{"stroke-width":str(a["sw"])}))
            elif k=="ellipse":
                d=common|dict(cx=str(a["x"]+a["w"]/2),cy=str(a["y"]+a["h"]/2),rx=str(a["w"]/2),ry=str(a["h"]/2),fill=a["fill"],stroke=a["stroke"],**{"stroke-width":str(a["sw"])})
                if a["rotate"]:d["transform"]=f'rotate({a["rotate"]} {a["x"]+a["w"]/2} {a["y"]+a["h"]/2})'
                ET.SubElement(root,"ellipse",d)
        ET.indent(root)
        ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)

    def drawio(self,path):
        root=ET.Element("mxfile",host="app.diagrams.net",type="device",version="24.7.17")
        dia=ET.SubElement(root,"diagram",id=self.name,name=self.title)
        model=ET.SubElement(dia,"mxGraphModel",dx=str(self.w),dy=str(self.h),grid="1",gridSize="10",
                            guides="1",tooltips="1",connect="1",arrows="1",fold="1",page="1",
                            pageScale="1",pageWidth=str(self.w),pageHeight=str(self.h),math="0",shadow="0")
        r=ET.SubElement(model,"root")
        ET.SubElement(r,"mxCell",id="0")
        ET.SubElement(r,"mxCell",id="1",parent="0")
        for a in self.el:
            k=a["kind"]
            attrs=dict(id=a["id"],value=a.get("s",""),parent="1")
            if k=="line":
                attrs.update(edge="1",style=f'edgeStyle=none;rounded=0;html=0;strokeColor={a["stroke"]};strokeWidth={a["sw"]};endArrow={"block" if a["arrow"] else "none"};endFill=1;endSize=10;startArrow=none;dashed={int(a["dash"])};dashPattern=6 6;')
                cell=ET.SubElement(r,"mxCell",attrs)
                g=ET.SubElement(cell,"mxGeometry",relative="1",**{"as":"geometry"})
                ET.SubElement(g,"mxPoint",x=str(a["pts"][0][0]),y=str(a["pts"][0][1]),**{"as":"sourcePoint"})
                ET.SubElement(g,"mxPoint",x=str(a["pts"][-1][0]),y=str(a["pts"][-1][1]),**{"as":"targetPoint"})
                if len(a["pts"])>2:
                    arr=ET.SubElement(g,"Array",**{"as":"points"})
                    for x,y in a["pts"][1:-1]:ET.SubElement(arr,"mxPoint",x=str(x),y=str(y))
                continue
            attrs["vertex"]="1"
            if k=="text":
                w,h=a["width"],a["height"]
                x=a["x"]-(w/2 if a["anchor"]=="middle" else w if a["anchor"]=="end" else 0)
                y=a["y"]-h/2
                attrs["style"]=f'text;html=0;whiteSpace=wrap;align={"center" if a["anchor"]=="middle" else "right" if a["anchor"]=="end" else "left"};verticalAlign=middle;resizable=1;autosize=0;overflow=visible;spacing=0;fontFamily={FAMILY};fontSize={a["size"]};fontColor={a["color"]};fontStyle={int(a["bold"])};strokeColor=none;fillColor=none;'
            elif k=="polygon":
                xx,yy=zip(*a["pts"]);x,y=min(xx),min(yy);w,h=max(xx)-x,max(yy)-y
                sh=ET.Element("shape",name=a["id"],w="100",h="100",aspect="variable",strokewidth="inherit")
                bg=ET.SubElement(sh,"background");pp=ET.SubElement(bg,"path")
                for j,(px,py) in enumerate(a["pts"]):ET.SubElement(pp,"move" if j==0 else "line",x=str(100*(px-x)/max(w,1)),y=str(100*(py-y)/max(h,1)))
                ET.SubElement(pp,"close");fg=ET.SubElement(sh,"foreground");ET.SubElement(fg,"fillstroke")
                cp=zlib.compressobj(wbits=-15);code=base64.b64encode(cp.compress(ET.tostring(sh))+cp.flush()).decode()
                attrs["style"]=f'shape=stencil({code});html=0;fillColor={a["fill"]};strokeColor={a["stroke"]};strokeWidth={a["sw"]};'
            else:
                x,y,w,h=a["x"],a["y"],a["w"],a["h"]
                attrs["style"]=f'shape={"ellipse" if k=="ellipse" else "rectangle"};rounded={int(a.get("radius",0)>0)};arcSize=10;html=0;fillColor={a["fill"]};strokeColor={a["stroke"]};strokeWidth={a["sw"]};rotation={a.get("rotate",0)};'
            cell=ET.SubElement(r,"mxCell",attrs)
            ET.SubElement(cell,"mxGeometry",x=f"{x:g}",y=f"{y:g}",width=f"{w:g}",height=f"{h:g}",**{"as":"geometry"})
        ET.indent(root)
        ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)

    def save(self,render=True):
        out=HERE/"figures";out.mkdir(parents=True,exist_ok=True)
        self.svg(out/f"{self.name}.svg")
        self.drawio(out/f"{self.name}.drawio")
        (out/f"{self.name}.caption.txt").write_text(self.caption,encoding="utf-8")
        (out/f"{self.name}.scene.json").write_text(json.dumps(dict(name=self.name,title=self.title,width=self.w,height=self.h,caption=self.caption,elements=self.el),ensure_ascii=False,indent=2))
        if render:
            qa=HERE/"qa";qa.mkdir(exist_ok=True)
            conf=qa/"fonts.conf"
            conf.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include><dir>{FONT.parent}</dir><cachedir>{qa/"fontcache"}</cachedir></fontconfig>')
            env=os.environ.copy();env["FONTCONFIG_FILE"]=str(conf)
            env["INKSCAPE_PROFILE_DIR"]=str(qa/"inkscape")
            subprocess.run(["inkscape",str(out/f"{self.name}.svg"),"--export-type=png",f'--export-filename={out/f"{self.name}.png"}',"--export-width=2400"],check=True,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            subprocess.run(["inkscape",str(out/f"{self.name}.svg"),"--export-type=pdf",f'--export-filename={out/f"{self.name}.pdf"}'],check=True,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return out


def matrix(s,x,y,cols=7,rows=5,cw=12,ch=12,base=None,missing=False,selected=None):
    base=base or C["teal"]
    for r in range(rows):
        for c in range(cols):
            alpha=.16+.5*((r*3+c*5+2)%9)/8
            f=tint(base,alpha)
            if selected is not None:f=tint(base,.7) if c in selected else C["faint"]
            if missing and (r*cols+c)%19==7:
                f=C["white"]
                s.rect(x+c*cw,y+r*ch,cw-2,ch-2,f,C["line"],.8)
                s.line([(x+c*cw+2,y+r*ch+ch-4),(x+c*cw+cw-4,y+r*ch+2)],C["gray"],1)
            else:s.rect(x+c*cw,y+r*ch,cw-2,ch-2,f)
    return (cols*cw,rows*ch)


def input_xy(s,x,y,style=False,source=False):
    if source:
        matrix(s,x,y+13,cols=8,rows=1,cw=15,ch=26,base=C["teal"])
        s.text(x+59,y+65,"真实源谱",22,facts="F11")
        s.text(x+59,y+98,"Pils/Lager → Tripel",21,color=C["text"],facts="F11")
        return
    matrix(s,x,y,cols=7,rows=5,cw=14,ch=14,missing=True)
    s.text(x+46,y+91,"X",24,bold=True,facts="F01")
    if style:
        for j in range(3):
            s.rect(x+111,y+j*24+2,36,15,tint(C["orange"],.25+j*.23),radius=2)
        s.text(x+129,y+91,"s",24,bold=True,facts="F01")
    else:
        matrix(s,x+112,y,cols=3,rows=5,cw=12,ch=14,base=C["orange"])
        s.text(x+129,y+91,"Y",24,bold=True,facts="F01")


def tiny_tree(s,x,y,scale=1):
    for a,b in [((0,0),(-18,22)),((0,0),(18,22)),((-18,22),(-29,43)),((-18,22),(-7,43)),((18,22),(7,43)),((18,22),(29,43))]:
        s.line([(x+a[0]*scale,y+a[1]*scale),(x+b[0]*scale,y+b[1]*scale)],C["teal"],1.7)
    for a,b in [(0,0),(-18,22),(18,22),(-29,43),(-7,43),(7,43),(29,43)]:s.dot(x+a*scale,y+b*scale,3.8*scale)


def candidate_comparison(s,x,y,w=480):
    s.line([(x+8,y+48),(x+48,y+48),(x+48,y+8),(x+93,y+8)],C["line"],2)
    s.line([(x+48,y+48),(x+93,y+48)],C["line"],2)
    s.line([(x+48,y+48),(x+48,y+91),(x+93,y+91)],C["line"],2)
    tiny_tree(s,x+131,y-8,.7)
    s.line([(x+107+u*2,y+63-20*math.exp(-u*u/90)) for u in range(25)],C["teal"],2)
    matrix(s,x+109,y+82,cols=5,rows=2,cw=10,ch=10)
    # Parallel candidates share one fixed evaluation protocol. The motif does
    # not average methods or imply a separate post-selection validation set.
    for source_y,target_y in [(8,29),(48,48),(91,67)]:
        s.line([(x+171,y+source_y),(x+225,y+source_y),(x+225,y+target_y),(x+265,y+target_y)],C["line"],2)
    for yy,length,color in [(29,33,C["gray"]),(48,48,C["teal"]),(67,25,C["gray"])]:
        s.line([(x+273,y+yy),(x+273+length,y+yy)],color,3,facts="F02,F18")
        s.dot(x+273+length,y+yy,3.5,fill=color,facts="F02,F18")
    s.line([(x+331,y+48),(x+337,y+54),(x+347,y+41)],C["orange"],2.2,facts="F02,F18")
    s.arrow(x+352,y+48,x+376,y+48,color=C["teal"])
    s.rect(x+385,y+25,74,47,tint(C["orange"],.13),C["orange"],1.6,radius=4,facts="F02,F18")
    s.text(x+422,y+47,"f+",28,color=C["orange"],bold=True,facts="F02,F18,F23")
    s.text(x+123,y+127,"基线与新候选",23,facts="F02")
    s.text(x+299,y+127,"统一协议",23,facts="F02")
    s.text(x+424,y+127,"预测候选",23,color=C["orange"],facts="F18")


def frozen(s,x,y,w=68,h=44):
    s.rect(x,y,w,h,C["teal"],radius=4,facts="F02,F10,F14")
    s.text(x+w/2,y+h/2-1,"f★",28,color=C["white"],bold=True,facts="F02,F10,F14")
    # Small native lock motif denotes the identical fixed six-model interface.
    s.line([(x+w-9,y-7),(x+w-9,y-11),(x+w-7,y-14),(x+w-3,y-14),
            (x+w-1,y-11),(x+w-1,y-7)],C["teal"],1.6,facts="F02,F10,F14")
    s.rect(x+w-12,y-7,15,12,C["white"],C["teal"],1.4,radius=2,facts="F02,F10,F14")
    s.dot(x+w-4.5,y-1.5,1.2,C["teal"],facts="F02,F10,F14")


def fingerprint(s,x,y,w=155,h=85,vertical=False):
    vals=[.32,.72,.47,.88,.57,.37,.66]
    for i,v in enumerate(vals):
        yy=y+i*h/7
        s.line([(x,yy),(x+w,yy)],C["faint"],6)
        s.line([(x+v*w*.54,yy),(x+v*w*.54+w*.23,yy)],C["teal2"],3)
        s.dot(x+v*w*.54+w*.12,yy,4.5,fill=C["teal"])


def graph(s,x,y,scale=1):
    pts=[(15,35),(47,10),(68,52),(37,80),(95,17),(139,24),(164,63),(126,90),(179,12)]
    s.ellipse(x-7*scale,y-10*scale,95*scale,111*scale,C["pale"],None,rotate=-12)
    s.ellipse(x+104*scale,y+1*scale,89*scale,103*scale,tint(C["orange"],.12),None,rotate=13)
    for i,j in [(0,1),(0,3),(0,2),(1,2),(2,3),(1,4),(2,4),(4,5),(5,6),(5,7),(6,7),(5,8),(6,8)]:
        s.line([(x+pts[i][0]*scale,y+pts[i][1]*scale),(x+pts[j][0]*scale,y+pts[j][1]*scale)],C["teal2"] if i<4 and j<5 else C["gray"],1.6)
    for i,(a,b) in enumerate(pts):s.dot(x+a*scale,y+b*scale,6*scale,fill=C["teal"] if i<5 else C["orange"],stroke=C["white"],sw=1)


def resampling(s,x,y):
    for offset in [16,8,0]:
        s.rect(x+offset,y-offset,91,78,C["white"],C["line"],1.3)
    for i,v in enumerate([35,60,42,51,25]):s.rect(x+10+i*14,y+67-v,8,v,C["teal2"])
    s.line([(x+107,y+28),(x+132,y+28),(x+132,y+63),(x+108,y+63)],C["orange"],2,arrow=True)


def top3(s,x,y):
    for i,(val,name) in enumerate([(102,"a"),(78,"b"),(47,"c")]):
        yy=y+i*30
        s.text(x-15,yy,name,22,color=C["text"])
        s.line([(x,yy),(x+val,yy)],C["teal2"],2.5)
        s.dot(x+val,yy,6,fill=C["teal"] if i<2 else C["orange"])


def cdf(s,x,y,w=150,h=83,interval=True):
    s.line([(x,y),(x,y+h),(x+w,y+h)],C["gray"],1.4)
    pts=[(x,y+h),(x+20,y+h),(x+20,y+h*.56),(x+58,y+h*.56),(x+58,y+h*.37),(x+91,y+h*.37),(x+91,y+h*.18),(x+126,y+h*.18),(x+126,y),(x+w,y)]
    s.line(pts,C["teal"],2.4)
    if interval:
        s.line([(x+20,y+h),(x+20,y+h*.56)],C["orange"],5)
        s.line([(x+20,y+h*.56),(x+w*.68,y+h*.56)],C["orange"],1.4,dash=True)


def covariance(s,x,y,n=6,cell=10,base=None):
    base=base or C["teal"]
    for i in range(n):
        for j in range(n):
            a=.76 if i==j else .12+.38*math.exp(-abs(i-j)/1.7)
            s.rect(x+i*cell,y+j*cell,cell-1.5,cell-1.5,tint(base,a))


def sparse(s,x,y,cells=10,cell=17):
    chosen={1,4,7}
    for i in range(cells):s.rect(x+i*cell,y,cell-3,25,tint(C["orange"],.8) if i in chosen else C["white"],C["orange"] if i in chosen else C["line"],1.3)
    s.text(x+cells*cell/2,y+49,"少量坐标改动",23,facts="F13")


def geom_small(s,x,y,w=175,h=103):
    s.ellipse(x+45,y+6,110,80,C["faint"],C["line"],1.4,rotate=-23)
    s.ellipse(x+72,y+25,60,43,"none",C["line"],1.4,rotate=-23)
    s.poly([(x+65,y+h),(x+w,y+h),(x+w,y+2),(x+115,y+2)],tint(C["teal"],.14),C["teal2"],1.2)
    s.dot(x+12,y+75,6,fill=C["gray"])
    s.arrow(x+21,y+73,x+99,y+57,color=C["orange"],sw=2.5)
    s.dot(x+105,y+55,6,fill=C["teal"])
    s.line([(x+112,y+36),(x+120,y+44),(x+131,y+28)],C["teal"],2.3)


def checks(s,x,y,small=False):
    size=20 if small else 24
    labels=["化学上界","线性守卫","f★ 感官复核" if small else "非线性复核"]
    for j,t in enumerate(labels):
        yy=y+j*(31 if small else 43)
        s.ellipse(x,yy-10,20,20,C["pale"],None)
        s.line([(x+5,yy),(x+9,yy+4),(x+15,yy-5)],C["teal"],1.8)
        s.text(x+34,yy,t,size,anchor="start",facts="F13,F14")


def overall():
    caption="化学观测连接感官预测、稳定关联与风格设计。四条工作线按实际输入依赖并列；基线、语义融合与现代模型各自在同一固定五折协议下产生折外预测，部署接口 f+ 按这些折外预测的核心30维评分选定，本轮采用原始231列至16维目标表示的TabICL。带锁的 f★ 是生成感官与迁移复核共同使用的冻结六模型接口。图形为结构示意。"
    s=Scene("01_framework","化学观测驱动的感官建模与风格设计",1600,900,caption)
    for x,t in [(338,"观测输入"),(786,"建模机制"),(1350,"任务输出")]:s.text(x,39,t,26,color=C["text"],bold=True)
    s.line([(232,70),(1568,70)],C["line"],1.6)
    for yy in [255,445,635,825]:s.line([(30,yy),(1570,yy)],C["line"],1.2)
    labels=[("01","感官预测","F02"),("02","稳定关联","F04"),("03a","条件生成","F06"),("03b","受约束迁移","F11")]
    for i,(n,title,fact) in enumerate(labels):
        yy=100+i*190
        s.text(34,yy,n,24,color=C["orange"],bold=True,anchor="start")
        s.text(34,yy+45,title,29,bold=True,anchor="start",facts=fact)
        s.arrow(447,yy+62,495,yy+62,color=C["gray"],sw=2)
        s.arrow(1111,yy+62,1159,yy+62,color=C["gray"],sw=2)
    input_xy(s,268,115)
    s.text(342,237,"语义仅辅助训练",18,color=C["orange"],facts="F18")
    candidate_comparison(s,505,114)
    s.text(760,92,"折内拟合",21,color=C["text"],facts="F02")
    s.arrow(985,162,1100,162,color=C["teal"],sw=2)
    s.text(1042,133,"固定折评价",22,color=C["text"],facts="F02,F18,F23")
    fingerprint(s,1193,120,146,88)
    for k in range(7):
        s.rect(1394+k*23,140,19,31,tint(C["teal"],.08+.1*k),None)
        s.text(1403+k*23,185,str(k),18,color=C["text"])
    s.text(1260,225,"50 维连续值",23,facts="F01,F03")
    s.text(1470,225,"相对等级",23,facts="F03")
    matrix(s,268,305,cols=6,rows=5,cw=13,ch=14,missing=True)
    matrix(s,365,305,cols=3,rows=5,cw=10,ch=14,base=C["orange"])
    for j in range(3):s.rect(409,309+j*22,20,12,tint(C["orange"],.25+j*.23),radius=2,facts="F20")
    for x,t in [(306,"X"),(379,"Y"),(419,"s")]:s.text(x,396,t,24,facts="F04,F20")
    # Raw paired observations and observed style enter the refit/check block
    # directly; the graph contributes the separate fixed module definitions.
    s.dot(471,352,2.3,fill=C["gray"],facts="F04,F20")
    s.line([(471,352),(471,278),(811,278),(811,298)],C["gray"],1.3,arrow=True,facts="F04,F20")
    graph(s,513,299,.88)
    s.arrow(712,352,752,352,color=C["gray"])
    resampling(s,772,316)
    s.arrow(921,352,960,352,color=C["gray"])
    for r in range(4):
        for c in range(5):s.dot(985+c*21,326+r*18,4.6,fill=C["teal"] if (r+c)%3 else tint(C["orange"],.55))
    s.text(601,414,"共变网络",23,facts="F04")
    s.text(824,414,"重拟合与风格复核",23,facts="F05,F20")
    s.text(1022,414,"联合频率",23,facts="F05")
    top3(s,1215,318)
    graph(s,1392,313,.59)
    s.text(1260,414,"Top 3 证据分层",23,facts="F05,F20")
    s.text(1460,414,"模块稳定性",23,facts="F05")
    input_xy(s,268,495,style=True)
    cdf(s,515,500,135,77)
    s.arrow(675,542,716,542,color=C["gray"])
    covariance(s,745,510,6,11)
    covariance(s,828,492,6,11,base=C["orange"])
    s.arrow(918,542,960,542,color=C["gray"])
    matrix(s,983,500,cols=8,rows=5,cw=12,ch=15)
    s.text(584,604,"概率区间",23,facts="F07")
    s.text(821,604,"风格收缩",23,facts="F08")
    s.text(1030,604,"广义逆采样",23,facts="F09")
    matrix(s,1193,500,cols=7,rows=5,cw=14,ch=15)
    s.arrow(1300,540,1332,540,color=C["gray"])
    frozen(s,1343,518,60,42)
    s.arrow(1414,540,1443,540,color=C["gray"])
    fingerprint(s,1454,501,110,81)
    s.text(1245,604,"虚拟化学谱",23,facts="F10")
    s.text(1493,604,"感官指纹",23,facts="F10")
    input_xy(s,268,688,source=True)
    sparse(s,515,711,10,17)
    s.arrow(705,732,750,732,color=C["gray"])
    geom_small(s,771,675,180,116)
    s.arrow(973,732,1015,732,color=C["gray"])
    matrix(s,1033,694,cols=4,rows=5,cw=15,ch=15,base=C["orange"],selected={1,3})
    s.text(866,798,"联合约束",23,facts="F12,F13,F15")
    s.text(1060,798,"回到原尺度",23,facts="F16")
    checks(s,1192,699,small=True)
    s.arrow(1375,732,1413,732,color=C["gray"])
    matrix(s,1435,699,cols=7,rows=4,cw=16,ch=16,base=C["orange"],selected={1,4,6})
    s.text(1490,798,"逐字段方案",23,facts="F16")
    s.text(800,865,"结构示意  ·  250 款观测  ·  X：化学测量  /  Y：感官标签  /  s：风格  ·  f★：冻结六模型感官接口",23,color=C["text"],facts="F01,F02,F18")
    return s


def conditional():
    caption="概率区间编码重复值，共享结构支撑小样本风格生成。广义逆保留经验原子支持；风格条件抽样下，各原子的概率不必等于原始经验跳跃质量。图中曲线、矩阵色块及样本点均为机制示意。"
    s=Scene("02_conditional_mechanism","保留原子支持的条件化学谱生成",1600,690,caption)
    stages=[(40,"a","经验概率质量"),(435,"b","潜在条件矩"),(830,"c","跨风格收缩"),(1225,"d","广义逆采样")]
    for x,a,t in stages:
        s.text(x,38,a,29,color=C["orange"],bold=True,anchor="start")
        s.text(x+41,38,t,29,bold=True,anchor="start")
    for x in [401,796,1191]:s.line([(x,83),(x,435)],C["line"],1.4)
    # a: repeated observations and ECDF jump.
    for j,xx in enumerate([94,94,94,94,156,156,220,270,315]):
        yy=110-12*(j if xx==94 else j-4 if xx==156 else 0)
        s.dot(xx,yy,5.5,fill=C["orange"] if xx==94 else C["teal"])
    s.text(204,141,"重复观测值",23,facts="F07")
    cdf(s,75,182,260,155)
    s.text(64,175,"F",22,color=C["text"])
    s.text(347,345,"x",22,color=C["text"])
    s.line([(105,333),(130,333),(130,270),(105,270)],C["orange"],1.7)
    s.text(172,294,"ΔF",25,color=C["orange"],bold=True)
    s.text(205,397,"一个取值 ↔ 一段概率区间",22,facts="F07")
    s.arrow(364,246,423,246,color=C["gray"],sw=2.2)
    # b: exact standard-normal schematic; no fitted chemical is represented.
    xx=lambda v:457+(v+3)*49
    yy=lambda v:324-167*math.exp(-v*v/2)
    lo,hi=-.95,.50
    band=[(xx(lo),324)]+[(xx(lo+(hi-lo)*i/30),yy(lo+(hi-lo)*i/30)) for i in range(31)]+[(xx(hi),324)]
    s.poly(band,tint(C["orange"],.28))
    s.line([(xx(-3),324),(xx(3),324)],C["gray"],1.5)
    s.line([(xx(-3+6*i/100),yy(-3+6*i/100)) for i in range(101)],C["teal"],2.4)
    for v,t in [(lo,"a"),(hi,"b")]:
        s.line([(xx(v),324),(xx(v),yy(v))],C["orange"],1.6,dash=True)
        s.text(xx(v),347,t,24,color=C["orange"])
    den=.5*(math.erf(hi/math.sqrt(2))-math.erf(lo/math.sqrt(2)))
    mu=(math.exp(-lo*lo/2)-math.exp(-hi*hi/2))/math.sqrt(2*math.pi)/den
    s.dot(xx(mu),324,6,fill=C["teal"])
    s.text(604,120,"正态区间",23,facts="F07")
    s.text(604,397,"均值 m   ·   方差 v",24,facts="F07")
    s.arrow(759,246,817,246,color=C["gray"],sw=2.2)
    # c: local/pool covariance and mean shrinkage.
    covariance(s,847,128,6,14,base=C["orange"])
    covariance(s,1045,128,6,14)
    s.text(888,105,"本风格",23,facts="F08")
    s.text(1087,105,"共享结构",23,facts="F08")
    s.line([(888,224),(888,250),(971,250),(971,274)],C["orange"],2,arrow=True)
    s.line([(1087,224),(1087,250),(1006,250),(1006,274)],C["teal"],2,arrow=True)
    covariance(s,950,290,6,14)
    for i in range(6):s.rect(950+i*14,290+i*14,12.5,12.5,tint(C["orange"],.55),None)
    s.text(992,397,"中心与协方差共同收缩",22,facts="F08")
    s.arrow(1154,246,1212,246,color=C["gray"],sw=2.2)
    # d: draw in latent space and reconstruct tied measurements.
    s.ellipse(1261,96,240,93,C["faint"],C["line"],1.4,rotate=-13)
    for a,b in [(27,45),(55,31),(74,52),(99,39),(124,26),(136,61),(165,42),(197,31),(178,66),(109,71)]:
        s.dot(1267+a,99+b,4.5,fill=C["teal2"])
    s.arrow(1382,213,1382,248,color=C["gray"])
    s.text(1485,231,"F⁻¹ ∘ Φ",27,color=C["teal"],bold=True)
    for i in range(9):
        xx0=1259+i*28
        for j in range([3,3,3,1,2,2,1,1,2][i]):s.rect(xx0,278+j*26,19,19,tint(C["teal"],.4 if i<3 else .16+.065*i),None)
    s.text(1380,397,"原子支持被保留",23,facts="F09")
    # Bottom path is actual post-generation processing.
    s.line([(40,450),(1560,450)],C["line"],1.4)
    s.text(74,496,"谱重建",25,bold=True,anchor="start",facts="F10")
    matrix(s,260,485,cols=9,rows=5,cw=13,ch=15)
    s.text(318,591,"225 直接变量",23,facts="F06")
    s.text(411,522,"+",33,color=C["gray"])
    matrix(s,463,492,cols=5,rows=4,cw=13,ch=16,base=C["orange"])
    s.text(495,591,"5 派生量",23,facts="F10")
    s.text(589,522,"+",33,color=C["gray"])
    s.rect(644,492,24,62,C["gray"])
    s.text(657,591,"1 内标",23,facts="F10")
    s.arrow(715,522,790,522,color=C["gray"])
    matrix(s,826,485,cols=10,rows=5,cw=13,ch=15)
    s.text(892,591,"231 维化学谱",23,facts="F10")
    s.arrow(980,522,1060,522,color=C["gray"])
    frozen(s,1082,498,77,49)
    s.arrow(1181,522,1253,522,color=C["gray"])
    fingerprint(s,1289,487,191,81)
    s.text(1383,591,"50 维感官指纹",23,facts="F10")
    s.text(800,655,"结构示意   ·   先生成化学谱，再由冻结预测器输出感官",21,color=C["text"],facts="F10")
    return s


def transfer(optimizer_label="稀疏联合优化"):
    caption="在全坐标中选择少量可调字段；所选子空间内，目标距离椭球、Tripel 风格半空间与缺陷半空间共同限定候选。源点位于 D₀ 参照边界，要求域对应距离至少下降 10%，即 D≤0.9D₀。返回原尺度后逐项检查化学边界和冻结感官模型。二维几何为机制示意，不是实测投影。"
    s=Scene("03_transfer_mechanism","稀疏风格迁移的几何与原空间复核",1600,735,caption)
    for x,a,t in [(42,"a","全坐标支持选择"),(506,"b","联合满足几何条件"),(1120,"c","原尺度逐项复核")]:
        s.text(x,42,a,29,color=C["orange"],bold=True,anchor="start")
        s.text(x+41,42,t,29,bold=True,anchor="start")
    s.line([(473,85),(473,628)],C["line"],1.4)
    s.line([(1087,85),(1087,628)],C["line"],1.4)
    # a: real source and sparse support selector.
    s.text(247,111,"真实源谱 x₀",24,bold=True,facts="F11")
    matrix(s,82,151,cols=13,rows=1,cw=25,ch=37)
    s.arrow(247,211,247,246,color=C["gray"])
    s.text(340,228,"225 维秩映射",22,color=C["text"],facts="F12,F17")
    s.text(247,281,optimizer_label,25,bold=True,facts="F13,F17")
    for i in range(13):
        chosen=i in {2,5,9}
        s.rect(82+i*25,317,21,33,tint(C["orange"],.8) if chosen else C["white"],C["orange"] if chosen else C["line"],1.5)
        if chosen:s.text(92.5+i*25,333,"1",19,color=C["white"],bold=True)
    s.text(247,389,"δⱼ = 0   （未选字段）",23,facts="F13,F16")
    s.text(247,435,"min |S|",31,color=C["teal"],bold=True,facts="F17")
    s.line([(92,498),(401,498)],C["line"],1.4)
    s.rect(105,526,27,27,C["gray"])
    s.text(153,539,"内标固定",23,anchor="start",facts="F16")
    s.line([(105,584),(131,584)],C["orange"],3)
    s.text(153,584,"派生量源锚定重建",23,anchor="start",facts="F16")
    # b: stylized 2-D projection of the three simultaneous geometric demands.
    x,y,w,h=529,120,533,405
    s.line([(x+18,y+350),(x+18,y+13)],C["gray"],1.5)
    s.line([(x+18,y+350),(x+w-4,y+350)],C["gray"],1.5)
    # coordinate box is a schematic chemical/rank restriction.
    s.rect(x+44,y+27,445,298,"none",C["gray"],1.5)
    # Tripel class halfspace within a selected sparse coordinate subspace.
    s.poly([(x+195,y+325),(x+489,y+325),(x+489,y+27),(x+305,y+27)],tint(C["teal"],.08),C["teal2"],1.5,facts="F13")
    sx,sy=x+102,y+257
    tx,ty=x+271,y+174
    ax,ay=x+300,y+198
    rx,ry=x+468,y+49
    # A mathematically consistent target-distance contour through the source.
    angle=math.radians(-18);ca,sa=math.cos(angle),math.sin(angle)
    u=ca*(sx-tx)+sa*(sy-ty);v=-sa*(sx-tx)+ca*(sy-ty)
    erx=math.sqrt(u*u+(v/.675)**2);ery=.675*erx
    contour=[(tx+erx*math.cos(t)*ca-ery*math.sin(t)*sa,
              ty+erx*math.cos(t)*sa+ery*math.sin(t)*ca)
             for t in [2*math.pi*i/160 for i in range(160)]]
    # Joint domain = distance ellipse ∩ Tripel halfspace ∩ defect halfspace.
    # This is a schematic; the exact constants are solely drawing coordinates.
    rho=.10
    required_contour=[(tx+(px-tx)*(1-rho),ty+(py-ty)*(1-rho)) for px,py in contour]
    joint=clip_halfspace(required_contour,-298,-110,-(298*(x+195)+110*(y+325)))
    joint=clip_halfspace(joint,.5,1,760)
    s.poly(joint,tint(C["teal"],.23),None,facts="F13,F15")
    s.line(contour+contour[:1],C["line"],1.7,dash=True,facts="F12,F17")
    for scale in [.90,.30]:
        s.ellipse(tx-erx*scale,ty-ery*scale,2*erx*scale,2*ery*scale,
                  "none",C["teal2"] if scale==.9 else C["line"],1.6,rotate=-18)
    s.line([(630,445),(1018,251)],C["text"],1.6,dash=True,facts="F13")
    s.text(981,308,"缺陷边界",20,color=C["text"],facts="F13")
    s.text(629,174,"D₀",23,color=C["text"],facts="F12,F17")
    s.text(664,226,"0.9 D₀",22,color=C["teal"],facts="F17")
    # source, target and two candidates; diagram is explicitly schematic.
    s.dot(tx,ty,5,fill=C["ink"])
    s.text(tx-10,ty-30,"μ",25,color=C["ink"],facts="F12")
    s.dot(sx,sy,7,fill=C["white"],stroke=C["ink"],sw=2.3)
    s.text(sx-20,sy+29,"源",23,color=C["ink"],facts="F11")
    s.arrow(sx+12,sy-4,ax-12,ay+3,color=C["orange"],sw=3)
    s.dot(ax,ay,8,fill=C["teal"],stroke=C["white"],sw=2)
    s.text(ax+12,ay+33,"联合候选",23,color=C["teal"],facts="F15")
    s.line([(sx+9,sy-8),(rx-11,ry+11)],C["reject"],1.8,dash=True,arrow=True)
    s.line([(rx-7,ry-7),(rx+7,ry+7)],C["reject"],2.3)
    s.line([(rx-7,ry+7),(rx+7,ry-7)],C["reject"],2.3)
    s.text(x+402,y+4,"仅分类可达",21,color=C["reject"],facts="F15")
    s.text(x+393,y+291,"Tripel 区域",22,color=C["teal"],facts="F13")
    s.text(793,543,"稀疏子空间中的联合可行域",24,bold=True,facts="F13,F15")
    s.text(793,588,"距离椭球  ∩  风格 / 缺陷半空间",22,color=C["text"],facts="F13")
    # c: original-scale completion, frozen predictor and audit gate.
    s.text(1332,111,"逆映射与谱补齐",24,bold=True,facts="F16")
    matrix(s,1181,151,cols=11,rows=2,cw=26,ch=24,base=C["orange"],selected={2,5,9})
    s.text(1332,232,"未选值逐项保留",23,color=C["text"],facts="F16")
    s.arrow(1332,267,1332,303,color=C["gray"])
    checks(s,1190,337)
    frozen(s,1444,403,70,44)
    s.line([(1433,423),(1400,423)],C["teal"],1.8,arrow=True)
    s.arrow(1332,472,1332,507,color=C["gray"])
    matrix(s,1200,534,cols=8,rows=3,cw=19,ch=17,base=C["orange"],selected={1,4,6})
    s.line([(1391,555),(1405,569),(1430,537)],C["teal"],3)
    s.text(1332,612,"合格候选 / 保留源谱",23,facts="F15")
    s.line([(42,647),(1558,647)],C["line"],1.4)
    s.text(800,690,"几何示意   ·   全部约束与感官复核在完整特征空间执行",23,color=C["text"],facts="F12,F13,F14")
    return s


def validate(scenes):
    report=[]
    for s in scenes:
        errs=[]
        for a in s.el:
            if a["kind"]=="text":
                width=a["width"];left=a["x"]-(width/2 if a["anchor"]=="middle" else width if a["anchor"]=="end" else 0)
                bounds=(left,a["y"]-a["height"]/2,left+width,a["y"]+a["height"]/2)
            elif a["kind"] in {"line","polygon"}:
                xx,yy=zip(*a["pts"]);bounds=(min(xx),min(yy),max(xx),max(yy))
            else:bounds=(a["x"],a["y"],a["x"]+a["w"],a["y"]+a["h"])
            if bounds[0]<0 or bounds[1]<0 or bounds[2]>s.w or bounds[3]>s.h:errs.append({"id":a["id"],"bounds":bounds})
        p=HERE/"figures"/f"{s.name}.drawio"
        root=ET.parse(p).getroot();cells=root.findall(".//mxCell")
        count=sum(c.attrib.get("vertex")=="1" or c.attrib.get("edge")=="1" for c in cells)
        assert count==len(s.el),(count,len(s.el))
        svg=(HERE/"figures"/f"{s.name}.svg").read_text()
        assert "<image" not in svg and "data:image" not in svg
        assert not any("image=" in c.attrib.get("style","") for c in cells)
        report.append(dict(figure=s.name,elements=len(s.el),drawio_editable_cells=count,
            native_texts=sum(a["kind"]=="text" for a in s.el),native_lines=sum(a["kind"]=="line" for a in s.el),
            no_raster_embedding=True,boundary_issues=errs,sha256_svg=hashlib.sha256(svg.encode()).hexdigest()))
    (HERE/"qa/structural_qa.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if any(r["boundary_issues"] for r in report):raise RuntimeError("Diagram boundary issues: inspect qa/structural_qa.json")
    return report


def main():
    global PROJECT, FONT
    ap=argparse.ArgumentParser()
    ap.add_argument("--optimizer-label",default="稀疏联合优化")
    ap.add_argument("--no-render",action="store_true")
    ap.add_argument("--project-dir",type=Path,help="MathorCup_A root, containing assets/fonts")
    ap.add_argument("--font-path",type=Path,help="Override the CJK font used to measure and render text")
    args=ap.parse_args()
    if args.project_dir:
        PROJECT=args.project_dir.resolve()
    elif not PROJECT.exists():
        PROJECT=next((p for p in HERE.parents if (p/"assets/fonts/NotoSansCJKsc-Regular.otf").is_file()),PROJECT)
    FONT=args.font_path.resolve() if args.font_path else PROJECT/"assets/fonts/NotoSansCJKsc-Regular.otf"
    if not FONT.is_file():
        ap.error("CJK font not found; pass --project-dir or --font-path.")
    (HERE/"qa").mkdir(parents=True,exist_ok=True)
    scenes=[overall(),conditional(),transfer(args.optimizer_label)]
    for s in scenes:s.save(render=not args.no_render)
    combined=ET.Element("mxfile",host="app.diagrams.net",type="device",version="24.7.17")
    for s in scenes:
        combined.append(ET.parse(HERE/"figures"/f"{s.name}.drawio").getroot().find("diagram"))
    ET.indent(combined)
    ET.ElementTree(combined).write(HERE/"figures/all_mechanisms.drawio",encoding="utf-8",xml_declaration=True)
    print(json.dumps(validate(scenes),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
