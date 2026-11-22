import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;P=R.parents[1]/'deep/prediction/fusion_replay_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[P/'outer_predictions.npz',P/'fitted_weights.csv',R/'diagnose.py',R/'README.md']+list(R.glob('*.csv'))
before={str(p):sha(p) for p in files}
z=np.load(P/'outer_predictions.npz',allow_pickle=False);Y=z['Y'];sd=Y.std(0)
U=(z['raw231_y16_no_residual']-Y)/sd;V=(z['raw231_y50_native']-Y)/sd
wt=pd.read_csv(P/'fitted_weights.csv',float_precision='round_trip').set_index('outer_fold').sort_index()
W=np.column_stack([np.repeat(wt.weight_core_y16.to_numpy()[z['fold'],None],30,axis=1),np.repeat(wt.weight_other_y16.to_numpy()[z['fold'],None],20,axis=1)])
methods={'fixed_half_y16_native50':np.full_like(W,.5),'inner_groupwise_y16_native50':W}
reconstruct={k:float(np.max(np.abs(w*z['raw231_y16_no_residual']+(1-w)*z['raw231_y50_native']-z[k]))) for k,w in methods.items()}
groups={'all50':np.arange(50),'core30':np.arange(30),'other20':np.arange(30,50)}
errors=[];scaled_errors=[];primary_errors=[];boolean_errors=[]
def check(row,expected):
 for key,val in expected.items():
  if isinstance(val,(str,bool,np.bool_)):
   if row[key]!=val:boolean_errors.append((key,str(row[key]),str(val)))
  elif not (pd.isna(val) and pd.isna(row[key])):
   err=abs(float(row[key])-float(val));errors.append(err);scaled_errors.append(err/max(1.,abs(float(val))))
   if key!='half_fraction_of_envelope':primary_errors.append(err)
def values(u,v):
 a=np.mean(u*u);b=np.mean(v*v);c=np.mean(u*v);d=np.mean((u-v)**2);gap=abs(a-b);h=np.mean(((u+v)/2)**2);g=min(a,b)-h
 env=(max(d-gap,0)**2/(4*d)) if d else 0.
 return {'y16_nmse':a,'native50_nmse':b,'uncentered_cross_moment':c,'disagreement':d,'endpoint_gap':gap,'best_endpoint':'y16' if a<=b else 'native50','half_nmse':h,'disagreement_credit':d/4,'endpoint_penalty':gap/2,'half_gain_over_best':g,'half_gain_formula':d/4-gap/2,'some_constant_convex_mix_improves':d>gap,'half_improves':d>2*gap,'empirical_constant_weight_envelope':env,'half_fraction_of_envelope':g/env if env else np.nan}
for _,row in pd.read_csv(R/'group_geometry.csv',float_precision='round_trip').iterrows():check(row,values(U[:,groups[row.targets]],V[:,groups[row.targets]]))
T=pd.read_csv(R/'per_target_geometry.csv',float_precision='round_trip')
assert list(T.target)==list(z['targets'])
for j,row in T.iterrows():check(row,values(U[:,j],V[:,j]))
for file in ['frozen_method_accounting.csv','fold_group_accounting.csv']:
 for _,row in pd.read_csv(R/file,float_precision='round_trip').iterrows():
  js=groups[row.targets];ix=np.ones(len(Y),bool) if file.startswith('frozen') else z['fold']==int(row['fold'])
  u=U[ix][:,js];v=V[ix][:,js];w=methods[row.method][ix][:,js];a=np.mean(u*u);b=np.mean(v*v);anchor=min(a,b);l=np.mean((w*u+(1-w)*v)**2);j=np.mean(w*(1-w)*(u-v)**2);p=np.mean(w*u*u+(1-w)*v*v)-anchor
  check(row,{'best_endpoint':'y16' if a<=b else 'native50','best_endpoint_nmse':anchor,'fusion_nmse':l,'disagreement_credit':j,'signed_endpoint_penalty':p,'gain_over_best':anchor-l,'accounted_gain':j-p,'n':int(ix.sum())})
G=pd.read_csv(R/'group_geometry.csv',float_precision='round_trip').set_index('targets');A=pd.read_csv(R/'frozen_method_accounting.csv',float_precision='round_trip');F=pd.read_csv(R/'fold_group_accounting.csv',float_precision='round_trip')
counts={'no_constant_mix_improves':int((T.disagreement<=T.endpoint_gap).sum()),'some_mix_but_not_half':int(((T.disagreement>T.endpoint_gap)&(T.disagreement<=2*T.endpoint_gap)).sum()),'half_improves':int((T.disagreement>2*T.endpoint_gap).sum())}
anchor_notes=[]
for group in groups:
 row=A[(A.targets==group)&(A.method=='inner_groupwise_y16_native50')].iloc[0];fold=F[(F.targets==group)&(F.method==row.method)];fg=float(np.average(fold.gain_over_best,weights=fold.n));anchor_notes.append({'group':group,'pooled_gain':float(row.gain_over_best),'mean_fold_gain_with_foldwise_best_anchor':fg,'anchor_change':float(row.gain_over_best-fg)})
ver=json.loads((R/'verification.json').read_text());hash_errors=[]
for name,digest in ver['source_sha256'].items():
 if sha(Path(name))!=digest:hash_errors.append(name)
for name,digest in ver['outputs'].items():
 if sha(R/name)!=digest:hash_errors.append(name)
if sha(R/'diagnose.py')!=ver['script_sha256']:hash_errors.append('diagnose.py')
assert max(scaled_errors)<1e-12 and max(primary_errors)<1e-12 and not boolean_errors and not hash_errors and max(reconstruct.values())<1e-12
assert all(sha(Path(p))==h for p,h in before.items())
review={'status':'PASS','scope':'Independent read-only algebra/source audit and NumPy recomputation from frozen OOF and weights; no learner imports, model fits, weight fitting/search or new resampling.',
 'checks':{'some_constant_convex_mix_improves_iff_D_gt_abs_a_minus_b':True,'fixed_half_improves_iff_D_gt_2_abs_a_minus_b':True,'D_zero_and_zero_endpoint_degeneracies_valid':True,'empirical_envelope_scope_is_common_constant_weight_only':True,'variable_weight_signed_endpoint_term_correct':True,'all_four_csvs_independently_recomputed':True,'cached_predictions_reconstructed_from_frozen_weights':True,'pooled250_ddof0_variance_used_consistently':True,'source_and_recorded_output_hashes_match':True,'inputs_unchanged':True},
 'prediction_max_abs_error':reconstruct,'numeric_fields_checked':len(errors),'csv_numeric_max_abs_error':max(errors),'csv_scaled_error_max':max(scaled_errors),'csv_primary_loss_and_moment_max_abs_error':max(primary_errors),'numeric_tolerance':'absolute 1e-12 for losses/moments; scaled max(1,abs(value))*1e-12 for envelope ratios','categorical_boolean_mismatches':boolean_errors,'per_target_counts':counts,
 'readme_main_values':A.to_dict('records'),'fold_anchor_nonadditivity':anchor_notes,
 'documentation_notes':[{'severity':'low','location':'README final bullet describing fold_group_accounting.csv','text':'Each fold uses its own best complete endpoint as anchor; weighted loss, J and the raw weighted endpoint term aggregate, but foldwise best-anchored gain/P do not directly equal pooled best-anchored gain/P. Clarify the phrase “以便聚合一致”. This does not change any CSV or the suggested body paragraph.','suggested_text':'各折沿用同一250行评价方差；损失及J可直接加权合并，gain/P需在合并后按统一最佳端点重新锚定。'}],
 'scientific_findings':[],'source_sha256':before,'audit_script_sha256':sha(Path(__file__))}
(R/'independent_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
lines=['# 冻结融合互补诊断的独立审查','','结论：**PASS**。未发现理论条件、主模型损失分解、CSV数值或冻结数据身份错误。仅有一项补充文档的聚合口径说明建议。','','## 理论核验','','- 同一固定加权范数下，从较优端点出发，净收益为 t(D−δ)−Dt²。因此存在严格有益共同凸权重当且仅当D>δ；等权严格优于最佳端点当且仅当D>2δ。D=0、零误差端点等退化情况也处理正确。','- c<min(a,b)的等价条件与未中心化归一化阈值正确；中心化相关本身不足以决定收益。','- 主方案按折、组变化权重时，J−P恒等式仍成立，P是可为负的带符号端点项。当前聚合P为正，README没有把它普遍称作非负惩罚。','- 事后包络仅覆盖单个共同常数权重类，未用于选权、部署或限制折组变化模型。','','## 独立数值核验','',f'- 从冻结250×50外层OOF与原权重重建两种冻结融合，最大绝对误差为{max(reconstruct.values()):.3g}。',f'- 四份CSV共{len(errors)}个数值项独立复算，损失/矩项最大绝对差{max(primary_errors):.3g}。含极小包络分母的收益比例项最大绝对差{max(errors):.3g}、最大缩放差{max(scaled_errors):.3g}；类别与布尔字段一致。',f'- 逐目标分类为无有益常数融合6维、存在有益融合但等权未获益11维、等权已获益33维。','- 全部50维：等权相对最佳端点净增0.012218027；主分组融合净增0.012666982。核心下降与其余维度补偿的差分加权账目正确。','- 所有已记录输入、脚本与CSV哈希一致；审查前后输入未变。没有重拟合、选新权重、模型调用或新重采样。','','## 文档聚合说明建议','','fold_group_accounting.csv逐折用该折最佳完整端点作锚；pooled行用合并评价样本的最佳完整端点作锚。min操作不与加权平均交换，因此损失和J可以直接按样本数合并，gain/P需要统一锚点。','',f'例如主方案全部50维：逐折gain的加权均值为{anchor_notes[0]["mean_fold_gain_with_foldwise_best_anchor"]:.9f}，pooled gain为{anchor_notes[0]["pooled_gain"]:.9f}，相差{anchor_notes[0]["anchor_change"]:.9f}来自锚点切换。建议README把“以便聚合一致”补充为“损失及J可直接加权合并，gain/P需按统一端点重新锚定”。现有CSV和拟加入正文的段落无需因此改变。']
(R/'independent_review.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':review['status'],'numeric_fields_checked':len(errors),'max_error':max(errors),'counts':counts,'reconstruction':reconstruct},ensure_ascii=False))
