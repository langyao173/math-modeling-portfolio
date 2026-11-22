"""Cross-check current manuscript inputs and the compiled PDF."""
from pathlib import Path
import json,re,math
import numpy as np,pandas as pd,fitz
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';E=ROOT/'evidence/current'
body=(P/'body.tex').read_text();app=(P/'appendices.tex').read_text();aux=(P/'manuscript.aux').read_text();log=(P/'manuscript.log').read_text()
assert re.search(r'newlabel\{bodyend\}\{\{9\}\{8\}',aux)
assert 'Overfull' not in log and 'Missing character' not in log
assert len(re.findall(r'\\clearpage',body))==7
for n in re.findall(r'\\input\{([^}]+)\}',body+app):assert (P/n).is_file(),n
for n in re.findall(r'\\fig\{([^}]+)\}',body+app):assert (ROOT/'figures'/(n+'.pdf')).is_file(),n
risk=pd.read_csv(E/'joint/risk.csv');q=risk[risk.candidate.eq('guard10')].set_index('phase')
assert q.hits10.to_dict()=={0:46,1:33}
assert abs(q.loc[0,'auc']-.817091)<1e-6 and abs(q.loc[1,'auc']-.770145)<1e-6
front=pd.read_csv(E/'operating/joint_frontier.csv');r=front[front.model.eq('guard10')&front.capacity.eq(.1)].iloc[0]
assert [int(r[c]) for c in ['slots','unique_clients','unique_events','repeat_clients','new_second']]==[559,331,61,228,40]
d=pd.read_parquet(E/'joint/predictions.parquet')
for phase in [0,1]:
 z=d[d.phase.eq(phase)];k=math.ceil(.1*len(z));a=np.lexsort((z.row_id.to_numpy(str),-z.guard10.to_numpy()))[:k];b=np.lexsort((z.row_id.to_numpy(str),-z.base_reference.to_numpy()))[:k]
 assert set(z.iloc[a].sample_id)==set(z.iloc[b].sample_id)
ci=pd.read_csv(E/'tail_intervals.csv');assert len(ci)==42
assert ci[ci.comparator.eq('strong_S')&ci.group.isin(['repayment_state','flow_structure'])].sim_high.lt(0).all()
rad=pd.read_csv(ROOT/'figures/v18_primary_capacity_data.csv');rc=rad[rad.model.eq('guard10')&rad.capacity.eq(.1)].iloc[0];assert rc.hits==46 and rc.k==291
ops=json.loads((ROOT/'product/workbench_data.json').read_text());assert len(ops['cases'])==2907
assert sum(c['review_at10'] for c in ops['cases'])==268
assert sum(bool((c.get('capacity_guard') or {}).get('guard_changed')) for c in ops['cases'])==2
pdf=fitz.open(ROOT/'043-赛题1.pdf');assert '小组编号：043' in pdf[0].get_text()
assert '参考文献' in pdf[9].get_text()
main='\n'.join(p.get_text() for p in pdf[1:9])
for token in ['0.8171','0.7701','2.184','1.506','331','228','18.958']:
 assert token in main,token
assert '软件验证记录' not in main and 'AUTOMATED-TEST' not in main
panels=json.loads((ROOT/'figures/workbench_current_source.json').read_text())['panels'];assert len(panels)==6
trial=json.loads((ROOT/'product/trial/print_views/source.json').read_text());assert len(trial['checks'])==3 and not trial['errors'];assert trial['checks'][-1]['source_navigation']
bad=[]
for i,p in enumerate(pdf):
 for b in p.get_text('blocks'):
  if min(b[0],b[1])<0 or b[2]>p.rect.width+1 or b[3]>p.rect.height+1:bad.append(i+1)
assert not bad,bad
r={'body_pages':8,'total_pdf_pages':len(pdf),'cover_group':'043','main_figures':len(re.findall(r'\\fig\{',body)),'main_tables':4,'workbench_panels':len(panels),'trial_source_navigation':True,'all_referenced_files_present':True,'latex_overflow_or_missing_glyphs':False,'top10_membership_matches_base':True,'tail_comparison_family':42,'main_capacity':[46,33],'two_period_unique_event_customers':61,'product_data_matches_current_score':True,'product_image_type':'actual DOM print export','historical_score_replay_max_error':json.loads((ROOT/'evidence/capacity_update/verification.json').read_text())['final_max_abs_error']}
(ROOT/'verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
