"""Render the self-contained workbench from its label-free frozen operational payload."""
from pathlib import Path
import base64,json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=ROOT/'product';a=p/'assets';d=json.loads((p/'workbench_data.json').read_text());rows=d['cases']
    assert len(rows)==2907 and sum(r['review_at10'] for r in rows)==268
    assert sum(r['score_available'] for r in rows)==2866
    for r in rows:
        assert all(h['relative_month']<=r['anchor_month'] for h in r['account_history'])
        if r['score_available']:assert abs(r['quality_index']-100*(1-r['risk_probability']))<1e-12
        else:assert r['risk_probability'] is None and r['quality_index'] is None
    css=''
    for family,name in [('ReportCJK','ReportChartCJK.ttf'),('UILatin','WorkbenchLatin.ttf')]:
        css+="@font-face{font-family:"+family+";src:url(data:font/ttf;base64,"+base64.b64encode((ROOT/'assets/fonts'/name).read_bytes()).decode()+") format('truetype');font-display:swap;}"
    css+=(a/'workbench.css').read_text()+'\n'+(a/'pages.css').read_text()+'\n'+(a/'protection.css').read_text()
    js='\n'.join((a/n).read_text() for n in ['case_workflow.js','workbench.js','industrial.js','unified.js','pages.js','protection.js'])
    data=json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    html=(a/'workbench.html').read_text().replace('<!--STYLE-->','<style>'+css+'</style>').replace('<!--DATA-->','<script id="case-data" type="application/json">'+data+'</script>').replace('<!--SCRIPT-->','<script>'+js+'</script>')
    (p/'风险复核工作台.html').write_text(html)
    catalog={r['case_key']:{k:r[k] for k in ['case_key','customer_key','account_key','row_id','reference_epoch','anchor_month']} for r in rows}
    (p/'case_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':')))
    receipt=dict(customers=len(rows),scoreable=sum(r['score_available'] for r in rows),queue=sum(r['review_at10'] for r in rows),history_records_observed=sum(h['actual_limit'] is not None for r in rows for h in r['account_history']),routes=dict(Counter(r['operating_queue'] for r in rows)),score_identity='q and Q=100(1-q)',contains_future_outcomes=False,primary_batch_customers=2677,primary_quota=268)
    (p/'source_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print('Rendered',len(rows),'customer cases')
if __name__=='__main__':main()
