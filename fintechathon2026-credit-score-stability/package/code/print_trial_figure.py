"""Paper appendix: same-case operational views from the actual trial application."""
from pathlib import Path
import fitz,logging,json
from weasyprint import HTML
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'product/trial/print_views';logging.getLogger('weasyprint').setLevel(logging.ERROR)
docs={};heights={}
for arm in 'ABC':
 HTML(filename=str(P/f'{arm}.html')).write_pdf(P/f'{arm}.pdf')
 d=fitz.open(P/f'{arm}.pdf');docs[arm]=d
 heights[arm]=max(b[3] for b in d[0].get_text('blocks'))+12
 d[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5),clip=fitz.Rect(0,0,d[0].rect.width,heights[arm])).save(P/f'{arm}_proof.png')
# Native component widths preserve readable text at manuscript scale.
W=900;g=24;col=(W-g)/2
ha=heights['A']/675*W
hb=max(heights['B'],heights['C'])/(438*.75)*col
H=40+ha+45+hb+45
out=fitz.open();page=out.new_page(width=W,height=H)
page.insert_font(fontname='Report',fontfile=str(ROOT/'assets/fonts/ReportChartCJK.ttf'))
color=(.18,.26,.21)
page.insert_text((0,21),'a  账务核对 · 直接读取六期原始记录',fontname='Report',fontsize=18,color=color)
page.insert_text((W-185,21),'同一练习案件 / P19',fontname='Report',fontsize=13,color=(.47,.52,.47))
page.show_pdf_page(fitz.Rect(0,34,W,34+ha),docs['A'],0,clip=fitz.Rect(0,0,675,heights['A']))
y=34+ha+38
for arm,x,label in [('B',0,'b  趋势分析 · 比较余额与还款'),('C',col+g,'c  证据协同 · 解释读数并接续')]:
 page.insert_text((x,y),label,fontname='Report',fontsize=18,color=color)
 h=heights[arm]/(438*.75)*col
 page.show_pdf_page(fitz.Rect(x,y+13,x+col,y+13+h),docs[arm],0,clip=fitz.Rect(0,0,438*.75,heights[arm]))
# Common workflow uses exact protocol steps, not measured efficacy claims.
y=H-27;page.draw_line((0,y-16),(W,y-16),color=(.79,.83,.79),width=.6)
for i,t in enumerate(['领取与计时','核对原始来源','形成办理材料','独立复审','退回同案接续']):
 page.insert_text((i*180+8,y+4),f'{i+1:02d}  '+t,fontname='Report',fontsize=13,color=color)
out.save(ROOT/'figures/trial_operations.pdf');page.get_pixmap(matrix=fitz.Matrix(2,2)).save(ROOT/'figures/trial_operations.png');out.close()
for d in docs.values():d.close()
(ROOT/'figures/trial_operations_source.json').write_text(json.dumps({'same_case':'P19','panels':['raw ledger','account charts','evidence agent'],'source':str(P.relative_to(ROOT)),'type':'actual_DOM_component_export','renderer':'WeasyPrint','heights':heights},ensure_ascii=False,indent=2))
print({'figure_size':[W,H],'component_heights':heights})
