"""Six actual workbench pages in one vector paper figure (two columns, three rows)."""
from pathlib import Path
import logging,json
import fitz
from weasyprint import HTML
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'product/print_views';logging.getLogger('weasyprint').setLevel(logging.ERROR)
routes=['overview','queue','review','handling','agents','tasks']
for r in routes:
 HTML(filename=str(P/f'{r}.html')).write_pdf(P/f'{r}.pdf')
 d=fitz.open(P/f'{r}.pdf');q=d[0];bottom=min(1000,max(b[3] for b in q.get_text('blocks'))+15)
 q.get_pixmap(matrix=fitz.Matrix(.9,.9),clip=fitz.Rect(0,0,915,bottom)).save(P/f'{r}_proof.png');d.close()
out=fitz.open();page=out.new_page(width=1280,height=1656)
font=ROOT/'assets/fonts/ReportChartCJK.ttf';page.insert_font(fontname='Report',fontfile=str(font))
labels=['a  客群总览 · 分流与风险分布','b  风险队列 · 优先排序与任务分派','c  客户分析 · 账务变化与名单保护','d  证据办理 · 动作核验与材料保存','e  AI 协同 · 检索、计算与来源草稿','f  任务接续 · 岗位、时效与办理记录']
clips=[]
for i,(r,label) in enumerate(zip(routes,labels)):
 x=10+(i%2)*640;y=8+(i//2)*550
 page.insert_text((x,y+18),label,fontname='Report',fontsize=19,color=(.18,.26,.21))
 d=fitz.open(P/f'{r}.pdf');bottom=760
 target=fitz.Rect(x,y+32,x+620,y+32+620*bottom/915)
 page.show_pdf_page(target,d,0,clip=fitz.Rect(0,0,915,bottom));page.draw_rect(target,color=(.83,.87,.82),width=.6)
 clips.append({'route':r,'source_page':0,'clip':[0,0,915,bottom]});d.close()
out.save(ROOT/'figures/workbench_current.pdf');page.get_pixmap(matrix=fitz.Matrix(2,2)).save(ROOT/'figures/workbench_current.png');out.close()
(ROOT/'figures/workbench_current_source.json').write_text(json.dumps({'type':'actual_DOM_print_export','renderer':'WeasyPrint','browser_screenshot':False,'panels':clips,'layout':'two columns, three rows','operational_records':'local exercise, actual claim/save API'},ensure_ascii=False,indent=2));print('Six-panel vector workbench figure generated.')
