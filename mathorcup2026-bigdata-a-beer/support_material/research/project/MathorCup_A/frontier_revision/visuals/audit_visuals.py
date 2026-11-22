#!/usr/bin/env python3
"""Verify native editability, vector outputs, and actual PDF rendering."""
from pathlib import Path
import base64,json,re,zlib,xml.etree.ElementTree as ET
import fitz
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
qa=HERE/"qa";qa.mkdir(exist_ok=True)
report={"mechanisms":[],"quantitative":[]}
for p in sorted((HERE/"figures").glob("*.scene.json")):
    scene=json.loads(p.read_text());stem=p.name.replace(".scene.json","")
    svg=ET.parse(p.with_name(stem+".svg")).getroot()
    xml=ET.parse(p.with_name(stem+".drawio")).getroot()
    cells=xml.findall(".//mxCell")
    byid={c.get("id"):c for c in cells if c.get("vertex")=="1" or c.get("edge")=="1"}
    assert len(byid)==len(scene["elements"])
    stencils=0;stencil_vertices=0
    for a in scene["elements"]:
        c=byid[a["id"]];style=c.get("style","")
        assert "image=" not in style and "data:image" not in style
        if a["kind"]=="text":assert c.get("value")==a["s"]
        if a["kind"]=="line":
            assert c.get("edge")=="1"
            assert len(c.findall(".//mxPoint"))==len(a["pts"])
        if a["kind"]=="polygon":
            code=re.search(r"shape=stencil\(([^)]+)\)",style).group(1)
            native=ET.fromstring(zlib.decompress(base64.b64decode(code),-15))
            assert native.tag=="shape" and native.find(".//close") is not None
            paths=native.findall(".//path/*")
            vertices=sum(t.tag in {"move","line"} for t in paths)
            assert vertices==len(a["pts"])
            assert native.find(".//image") is None
            stencils+=1;stencil_vertices+=vertices
    assert not any(e.tag.endswith("}image") for e in svg.iter())
    pdf=fitz.open(p.with_name(stem+".pdf"));assert len(pdf)==1
    page=pdf[0];assert not page.get_images(full=True)
    assert any("\u4e00"<=t<="\u9fff" for t in page.get_text())
    page.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(qa/(stem+"_pdf.png"))
    report["mechanisms"].append(dict(name=stem,native_cells=len(byid),
        native_text=sum(a["kind"]=="text" for a in scene["elements"]),
        native_edges=sum(a["kind"]=="line" for a in scene["elements"]),
        native_closed_stencils=stencils,stencil_path_vertices=stencil_vertices,
        raster_embedding=False,xml_and_scene_match=True,pdf_embedded_images=0,
        pdf_text_characters=len(page.get_text())))

combined=ET.parse(HERE/"figures/all_mechanisms.drawio").getroot()
pages=combined.findall("diagram")
assert len(pages)==len(report["mechanisms"])==3
for page,item in zip(pages,report["mechanisms"]):
    single=ET.parse(HERE/"figures"/(item["name"]+".drawio")).getroot().find("diagram/mxGraphModel")
    merged=page.find("mxGraphModel")
    assert ET.tostring(merged)==ET.tostring(single)
report["combined_drawio"]={"pages":3,"matches_individual_graph_models":True,
    "native_cells":sum(r["native_cells"] for r in report["mechanisms"])}

previews=[]
for p in sorted((HERE/"quantitative").glob("*.pdf")):
    doc=fitz.open(p);assert len(doc)==1
    page=doc[0];text=page.get_text();fonts=page.get_fonts(full=True)
    assert not page.get_images(full=True)
    assert any("\u4e00"<=c<="\u9fff" for c in text)
    pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
    out=qa/(p.stem+"_pdf.png");pix.save(out)
    previews.append(out)
    svg=ET.parse(p.with_suffix(".svg")).getroot()
    assert not any(e.tag.endswith("}image") for e in svg.iter())
    report["quantitative"].append(dict(name=p.stem,pdf_pages=1,pdf_embedded_images=0,
        text_characters=len(text),font_count=len(fonts),native_svg_text=sum(e.tag.endswith("}text") for e in svg.iter()),
        page_points=[page.rect.width,page.rect.height],pdf_preview=str(out)))

tiles=[]
for p in previews:
    im=Image.open(p).convert("RGB")
    im.thumbnail((1000,650))
    tile=Image.new("RGB",(1040,690),"white");tile.paste(im,((1040-im.width)//2,40))
    ImageDraw.Draw(tile).text((20,15),p.stem,fill="#263B43")
    tiles.append(tile)
contact=Image.new("RGB",(2080,690*((len(tiles)+1)//2)),"#EEF1F2")
for i,im in enumerate(tiles):contact.paste(im,((i%2)*1040,(i//2)*690))
contact.save(qa/"quantitative_pdf_contact.png")
(qa/"native_vector_and_pdf_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
