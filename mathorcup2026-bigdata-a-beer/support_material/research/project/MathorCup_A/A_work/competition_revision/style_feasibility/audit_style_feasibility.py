"""Independently rebuild constraints and audit persisted witnesses and candidates.

No result from the search is treated as valid solely because a solver flag says
success. Infeasible cases are checked by a nonnegative weighted halfspace sum
whose minimum over the coordinate box is strictly positive.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time,ast,csv
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;AWORK=HERE.parents[1]
sys.path.insert(0,str(AWORK/'q3'));sys.path.insert(0,str(AWORK/'q1'))
from conditional_model import TECHNICAL
from transfer_profiles import DEFECTS
from sensory_model import predict_bundle
EPS=1e-7;TOL=1e-8

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def csv_tokens(path):
 """Read serialized tokens without numeric coercion or missing-value rewriting."""
 with Path(path).open(encoding='utf-8-sig',newline='') as stream:
  rows=list(csv.reader(stream))
 if not rows or len(set(rows[0]))!=len(rows[0]) or any(len(row)!=len(rows[0]) for row in rows[1:]):
  raise ValueError(f'Invalid rectangular CSV with unique columns: {path}')
 return rows[0],rows[1:]

def audit_old_profile_usage(features):
 """Check every old_profiles name use, and the actual classifier training inputs."""
 path=HERE/'run_style_feasibility.py';source=path.read_text();tree=ast.parse(source)
 parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
 statements={}
 for node in ast.walk(tree):
  if isinstance(node,ast.Name) and node.id=='old_profiles':
   statement=node
   while not isinstance(statement,ast.stmt):statement=parents[statement]
   statements[statement.lineno]=statement
 observed=[statements[k] for k in sorted(statements)]
 expected=ast.parse("""old_profiles=pd.read_csv(files['old_profiles'])
mask=old_profiles.method.str.startswith('forward_schur_k')|old_profiles.method.eq('all_225')
old_profiles=old_profiles.loc[mask].reset_index(drop=True)
old_diag=old_diag.set_index(['beer','method']).loc[list(zip(old_profiles.beer,old_profiles.method))].reset_index()
old_z=control.map_.transform(old_profiles[features].to_numpy(float))
""").body
 dump=lambda node:ast.dump(node,include_attributes=False)
 assignments={}
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign):
   for target in node.targets:
    if isinstance(target,ast.Name):assignments.setdefault(target.id,[]).append(node)
 expected_inputs={
  'df':"df=pd.read_csv(files['data'])",
  'features':"features=list(pack['features'])",
  'x':"x=df[features].to_numpy(float)",
  'z':"z=control.map_.transform(x)",
  'clf':"clf=LogisticRegression(C=.1,max_iter=3000,class_weight='balanced').fit(z,df.tasting_category_fine)"}
 input_checks={name:len(assignments.get(name,[]))==1 and dump(assignments[name][0])==dump(ast.parse(code).body[0]) for name,code in expected_inputs.items()}
 excluded={'tasting_category_fine','source_tasting_category_fine','requested_target_style','diagnostic_predicted_style','profile_origin','tasting_category_fine_origin'}
 checks=dict(all_old_profiles_uses_match_restricted_statements=[dump(node) for node in observed]==[dump(node) for node in expected],
  original_data_and_frozen_feature_training_inputs=all(input_checks.values()),
  category_metadata_excluded_from_225_numeric_features=len(features)==225 and not bool(excluded.intersection(features)))
 return dict(passed=all(checks.values()),checks=checks,training_input_checks=input_checks,
  old_profiles_usage=[dict(line=node.lineno,statement=ast.get_source_segment(source,node)) for node in observed],
  classifier_training_statement=ast.get_source_segment(source,assignments['clf'][0]) if len(assignments.get('clf',[]))==1 else None,
  interpretation='old_profiles supplies only method filtering, beer/method alignment and the frozen numeric feature matrix; classifier labels come from original df.tasting_category_fine, not exported profile metadata')

def audit_documented_metadata_changes(protocol,current_inputs,hash_match,features):
 """Accept only fully verified export-metadata patches; retain raw hash failures."""
 verification=HERE.parent/'verification';reports={}
 new_fields=['source_tasting_category_fine','requested_target_style','diagnostic_predicted_style','profile_origin','tasting_category_fine_origin']
 if not hash_match['old_profiles']:
  result={'passed':False};reports['old_profiles']=result
  try:
   ledger_path=verification/'metadata_patch.json';backup=verification/'selected_profiles_before_metadata.csv'
   ledger=json.loads(ledger_path.read_text());actual_before=sha256(backup);actual_after=sha256(current_inputs['old_profiles'])
   old_columns,old_rows=csv_tokens(backup);columns,rows=csv_tokens(current_inputs['old_profiles'])
   source_columns,source_rows=csv_tokens(current_inputs['data']);diagnostic_columns,diagnostic_rows=csv_tokens(current_inputs['old_diagnostics'])
   old_records=[dict(zip(old_columns,row)) for row in old_rows];records=[dict(zip(columns,row)) for row in rows]
   source_records=[dict(zip(source_columns,row)) for row in source_rows];diagnostic_records=[dict(zip(diagnostic_columns,row)) for row in diagnostic_rows]
   source_labels={row['beer']:row['tasting_category_fine'] for row in source_records}
   diagnostics={(row['beer'],row['method']):row for row in diagnostic_records}
   same_shape=len(rows)==len(old_rows) and columns==old_columns+new_fields
   comparable=[column for column in old_columns if column!='tasting_category_fine']
   mismatch_count=(sum(old[column]!=new.get(column) for old,new in zip(old_records,records) for column in comparable) if len(rows)==len(old_rows) else None)
   source_keys_complete=all(row['beer'] in source_labels for row in records)
   diagnostic_keys_complete=all((row['beer'],row['method']) in diagnostics for row in records)
   usage=audit_old_profile_usage(features)
   checks=dict(protocol_expected_equals_ledger_before=protocol['input_sha256']['old_profiles']==ledger['file_sha256_before'],
    backup_hash_equals_ledger_before=actual_before==ledger['file_sha256_before'],current_hash_equals_ledger_after=actual_after==ledger['file_sha256_after'],
    ledger_new_fields_exactly_specified=ledger['new_fields']==new_fields,old_columns_retained_in_order_and_only_five_fields_added=same_shape,
    all_preexisting_noncategory_csv_tokens_identical=same_shape and mismatch_count==0,
    source_and_diagnostics_original_hashes_match=hash_match['data'] and hash_match['old_diagnostics'],
    source_beer_keys_unique=len(source_labels)==len(source_records),diagnostic_beer_method_keys_unique=len(diagnostics)==len(diagnostic_records),
    source_keys_complete=source_keys_complete,diagnostic_keys_complete=diagnostic_keys_complete,
    observed_source_category_correct=source_keys_complete and all(row.get('tasting_category_fine')==source_labels[row['beer']] and row.get('source_tasting_category_fine')==source_labels[row['beer']] for row in records),
    requested_target_style_correct=all(row.get('requested_target_style')=='Tripel' for row in records),
    diagnostic_predicted_style_correct=diagnostic_keys_complete and all(row.get('diagnostic_predicted_style')==diagnostics[(row['beer'],row['method'])]['predicted_style'] for row in records),
    nontrivial_flags_are_boolean=diagnostic_keys_complete and all(diagnostics[(row['beer'],row['method'])]['nontrivial'] in ['True','False'] for row in records),
    profile_origin_correct=diagnostic_keys_complete and all(row.get('profile_origin')==('virtual_transfer' if diagnostics[(row['beer'],row['method'])]['nontrivial']=='True' else 'observed_source_fallback') for row in records),
    category_origin_correct=all(row.get('tasting_category_fine_origin')=='observed_source_label; transfer_not_remeasured' for row in records),
    old_profile_metadata_not_used_in_optimization=usage['passed'])
   result.update(passed=all(checks.values()),checks=checks,ledger=str(ledger_path),backup=str(backup),
    protocol_expected_before=protocol['input_sha256']['old_profiles'],actual_backup_sha256=actual_before,actual_current_sha256=actual_after,
    old_rows=len(old_rows),new_rows=len(rows),old_columns=len(old_columns),new_columns=len(columns),
    compared_noncategory_tokens=len(old_rows)*len(comparable),changed_noncategory_tokens=mismatch_count,
    changed_observed_category_tokens=sum(old['tasting_category_fine']!=new.get('tasting_category_fine') for old,new in zip(old_records,records)),
    added_fields=new_fields,usage_evidence=usage)
  except Exception as error:result['error']=f'{type(error).__name__}: {error}'
 if not hash_match['reference_script']:
  result={'passed':False};reports['reference_script']=result
  try:
   ledger_path=verification/'metadata_export_code_patch.json';backup=verification/'run_budget_transfer_before_metadata.py'
   ledger=json.loads(ledger_path.read_text());before=backup.read_bytes();after=current_inputs['reference_script'].read_bytes()
   removed=['    selected_profiles["tasting_category_fine"] = np.where(selected.nontrivial, "Tripel", "Pils/Lager")']
   added=['    # Metadata only: the supplied category describes the observed source.',
    '    # A requested destination and a fitted classifier are not new tasting labels.',
    '    selected_profiles["source_tasting_category_fine"] = selected_profiles.beer.map(original.tasting_category_fine)',
    '    selected_profiles["tasting_category_fine"] = selected_profiles.source_tasting_category_fine',
    '    selected_profiles["requested_target_style"] = "Tripel"',
    '    selected_profiles["diagnostic_predicted_style"] = selected.predicted_style.to_numpy()',
    '    selected_profiles["profile_origin"] = np.where(selected.nontrivial, "virtual_transfer", "observed_source_fallback")',
    '    selected_profiles["tasting_category_fine_origin"] = "observed_source_label; transfer_not_remeasured"']
   removed_bytes='\n'.join(ledger['removed_lines']).encode();added_bytes='\n'.join(ledger['added_lines']).encode()
   actual_before=hashlib.sha256(before).hexdigest();actual_after=hashlib.sha256(after).hexdigest()
   checks=dict(protocol_expected_equals_ledger_before=protocol['input_sha256']['reference_script']==ledger['before_sha256'],
    backup_hash_equals_ledger_before=actual_before==ledger['before_sha256'],current_hash_equals_ledger_after=actual_after==ledger['after_sha256'],
    removed_lines_exactly_expected_export_assignment=ledger['removed_lines']==removed,
    added_lines_exactly_expected_export_metadata=ledger['added_lines']==added,removed_block_occurs_exactly_once=before.count(removed_bytes)==1,
    complete_code_equals_backup_with_only_documented_replacement=after==before.replace(removed_bytes,added_bytes))
   result.update(passed=all(checks.values()),checks=checks,ledger=str(ledger_path),backup=str(backup),
    protocol_expected_before=protocol['input_sha256']['reference_script'],actual_backup_sha256=actual_before,actual_current_sha256=actual_after,
    complete_code_bytes_compared=len(after),removed_lines=removed,added_lines=added)
  except Exception as error:result['error']=f'{type(error).__name__}: {error}'
 return reports

def main():
 t=time.perf_counter();protocol=json.loads((HERE/'protocol.json').read_text())
 data=Path(os.environ.get('MATHORCUP_DATA_DIR',AWORK.parent/'data/A')).expanduser().resolve()
 df=pd.read_csv(data/'1.csv');src=df.set_index('beer')
 pack=joblib.load(AWORK/'q3/transfer_and_augmentation_models.joblib');model=pack['model'];features=pack['features'];a=pack['stale_proxy'].coef_
 q1=joblib.load(AWORK/'q1/sensory_ensemble.joblib');clf=joblib.load(HERE/'frozen_style_classifier.joblib')
 lp=pd.read_csv(HERE/'linear_feasibility.csv');supports=pd.read_csv(HERE/'support_definitions.csv').set_index('case_id')
 witnesses=np.load(HERE/'linear_witnesses_and_certificates.npz');deltas=witnesses['delta'];dual=witnesses['dual_weights']
 ti=list(clf.classes_).index('Tripel');others=np.flatnonzero(np.arange(len(clf.classes_))!=ti)
 ds=clf.coef_[ti]-clf.coef_[others];db=clf.intercept_[ti]-clf.intercept_[others]
 contexts={}
 for beer in lp.beer.unique():
  x0=src.loc[beer,features].to_numpy(float);z0=model.map_.transform(x0[None])[0]
  lower=model.map_.lower_.copy();upper=model.map_.upper_.copy();caps={}
  for col,info in DEFECTS.items():
   j=features.index(col);cap=x0[j] if np.isfinite(x0[j]) else (np.log10(info['threshold']) if info['transform']=='log10' else info['threshold'])
   row=model.map_.impute(x0[None])[0];row[j]=cap
   upper[j]=min(upper[j],model.map_.transform(row[None])[0,j]);caps[j]=cap
  contexts[int(beer)]=(x0,z0,lower,upper,caps)
 lp_audit=[];ipm_audit=[]
 for r in lp.itertuples():
  x0,z0,lower,upper,caps=contexts[r.beer];sel=np.array([features.index(c) for c in json.loads(supports.loc[r.case_id,'selected_fields'])]);frozen=np.ones(len(features),bool);frozen[sel]=False
  dd=deltas[r.lp_id];A=np.vstack([a[sel],-ds[:,sel]]);rhs=np.r_[-r.margin,ds@z0+db-EPS];lo=lower[sel]-z0[sel];hi=upper[sel]-z0[sel]
  score=clf.decision_function((z0+dd)[None])[0];stylemargin=float(np.min(score[ti]-score[others]));lin=float(a@dd)
  boundvio=max(float(np.max(lo-dd[sel])),float(np.max(dd[sel]-hi)))
  frozenvio=float(np.max(np.abs(dd[frozen]),initial=0.))
  rec=dict(lp_id=r.lp_id,beer=r.beer,method=r.method,margin=r.margin,reported_feasible=r.lp_feasible,bound_violation=boundvio,frozen_delta_violation=frozenvio,min_style_margin=stylemargin,linear_stale_delta=lin)
  if r.lp_feasible:
   violation=max(boundvio,float(np.max(A@dd[sel]-rhs)),frozenvio)
   ok=bool(violation<=TOL and stylemargin>=EPS-TOL and lin<=-r.margin+TOL)
   rec.update(pass_audit=ok,full_constraint_violation=violation,independent_separation_lower_bound=np.nan)
  else:
   weights=dual[r.lp_id];slope=weights@A
   certificate=float(np.sum(np.where(slope>=0,slope*lo,slope*hi))-weights@rhs)
   # Conservative floating arithmetic scale estimate, not a symbolic certificate.
   arithmetic_scale=float(np.sum(np.abs(slope)*np.maximum(np.abs(lo),np.abs(hi)))+np.sum(np.abs(weights*rhs)))
   roundoff_allowance=1e-12*max(1.,arithmetic_scale)
   phase_violation=float(np.max(A@dd[sel]-r.phase1_min_relaxation-rhs))
   ok=bool(np.min(weights)>=0 and abs(weights.sum()-1)<1e-8 and certificate>roundoff_allowance and boundvio<=TOL and phase_violation<=TOL and frozenvio<=TOL)
   rec.update(pass_audit=ok,independent_separation_lower_bound=certificate,separation_roundoff_allowance=roundoff_allowance,phase1_max_relaxed_violation=phase_violation)
  lp_audit.append(rec)
  if r.margin==0.:
   alt=linprog(np.zeros(len(sel)),A_ub=A,b_ub=rhs,bounds=list(zip(lo,hi)),method='highs-ipm',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
   alt_vio=(max(float(np.max(lo-alt.x)),float(np.max(alt.x-hi)),float(np.max(A@alt.x-rhs))) if alt.x is not None else np.nan)
   alt_feas=bool(alt.success and alt_vio<=TOL)
   ipm_audit.append(dict(case_id=r.case_id,beer=r.beer,method=r.method,highs_ipm_status=int(alt.status),highs_ipm_feasible=alt_feas,highs_ipm_residual=alt_vio,agrees_with_original=alt_feas==r.lp_feasible))
 lp_audit=pd.DataFrame(lp_audit);ipm_audit=pd.DataFrame(ipm_audit)
 lp_audit.to_csv(HERE/'audit_lp_witnesses.csv',index=False);ipm_audit.to_csv(HERE/'audit_independent_lp_method.csv',index=False)
 # round_trip recovers exactly the float values serialized by the search.
 final=pd.read_csv(HERE/'selected_profiles.csv',float_precision='round_trip');diag=pd.read_csv(HERE/'selected_diagnostics.csv',float_precision='round_trip')
 pred=predict_bundle(q1,final);basepred=predict_bundle(q1,df);basepred.index=df.beer
 mu=model.mean_['Tripel'];precision=np.linalg.inv(model.cov_['Tripel'])
 ZZ=model.map_.transform(final[features].to_numpy(float));pred_style=clf.predict(ZZ);pred_prob=clf.predict_proba(ZZ)[:,ti]
 candidate_audit=[]
 for i,row in final.iterrows():
  beer=int(row.beer);x0,z0,lower,upper,caps=contexts[beer];zn=ZZ[i];dd=zn-z0
  record=diag.iloc[i];assert row.beer==record.beer and row.method==record.method
  chosen=supports[(supports.beer==beer)&supports.method.eq(row.method)].iloc[0];selected=set(json.loads(chosen.selected_fields));same=[];frozen_equal=True
  for j,col in enumerate(features):
   eq=bool((pd.isna(x0[j]) and pd.isna(row[col])) or x0[j]==row[col]);same.append(eq)
   if col not in selected:frozen_equal &= eq
  tech_equal=all(bool((pd.isna(src.loc[beer,c]) and pd.isna(row[c])) or src.loc[beer,c]==row[c]) for c in TECHNICAL)
  raw_changed=int(np.sum(~np.asarray(same)));rank_changed=int(np.sum(np.abs(dd)>TOL));score=clf.decision_function(zn[None])[0];margin=float(np.min(score[ti]-score[others]))
  boundvio=max(float(np.max(lower-zn)),float(np.max(zn-upper)))
  chemvio=max([float(row[features[j]]-cap) for j,cap in caps.items() if np.isfinite(row[features[j]])]+[0.]);missing=sum(not np.isfinite(row[features[j]]) for j in caps)
  linear=float(a@dd);nonlinear=float(pred.loc[i,'stale_hops']-basepred.loc[beer,'stale_hops'])
  d0=float(np.sqrt((z0-mu)@precision@(z0-mu)/len(features)));dn=float(np.sqrt((zn-mu)@precision@(zn-mu)/len(features)));obj=float(.5*(zn-mu)@precision@(zn-mu)+.1*(dd@dd))
  accepted=bool(margin>=EPS-TOL and pred_style[i]=='Tripel' and boundvio<=TOL and chemvio<=TOL and missing==0 and linear<=TOL and nonlinear<=TOL)
  auxiliary_error=0.;kcal_error=0.
  if not row.source_fallback:
   for col,reg in pack['derived_predictors'].items():
    before=src.loc[beer,col];now=float(reg.predict(zn[None])[0]);old=float(reg.predict(z0[None])[0]);expected=before+now-old if np.isfinite(before) else now
    expected=np.clip(expected,np.nanmin(df[col]),np.nanmax(df[col]));auxiliary_error=max(auxiliary_error,abs(expected-row[col]))
   kcal=np.log10(5.523*10.**row['ethanol..v.v.']+.4*(10.**row['GFS']+10.**row['unfermentables']+10.**row['protein.g.L.'])+.0003*10.**row['lactic_acid.mg.L.'])
   kcal_error=abs(kcal-row.kcalperc)
  metric_error=max(abs(dn-record.target_distance),abs(1-dn/d0-record.target_distance_reduction),abs(obj-record.quadratic_objective),abs(nonlinear-record.ensemble_stale_delta),abs(linear-record.linear_stale_delta),abs(pred_prob[i]-record.target_probability),abs(margin-record.min_style_margin))
  ok=bool(frozen_equal and tech_equal and raw_changed<=row.allowed_coordinates and raw_changed==record.raw_direct_fields_changed and rank_changed==record.rank_coordinates_changed and accepted==record.accepted and auxiliary_error<1e-10 and kcal_error<1e-12 and metric_error<1e-8)
  candidate_audit.append(dict(beer=beer,method=row.method,allowed_coordinates=row.allowed_coordinates,source_fallback=row.source_fallback,accepted=accepted,pass_audit=ok,
    frozen_raw_exact=frozen_equal,internal_standard_exact=tech_equal,raw_direct_fields_changed=raw_changed,rank_coordinates_changed=rank_changed,
    min_style_margin=margin,max_rank_bound_violation=boundvio,chemical_violation=chemvio,unresolved_missing=missing,linear_stale_delta=linear,nonlinear_stale_delta=nonlinear,
    auxiliary_max_error=auxiliary_error,kcal_error=kcal_error,recomputed_metric_max_error=metric_error))
 ca=pd.DataFrame(candidate_audit);ca.to_csv(HERE/'audit_selected_candidates.csv',index=False)
 monotonic=[]
 for beer,g in lp[lp.margin==0].groupby('beer'):
  values=g.sort_values('allowed_coordinates').lp_feasible.to_numpy(int);monotonic.append(bool(np.all(np.diff(values)>=0)))
 # Resolve the recorded semantic inputs inside the current package/data root;
 # the original content hashes remain the evidence, not historical locations.
 current_inputs={
  'data':data/'1.csv','calibrated':data/'8.xlsx',
  'frozen_q3':AWORK/'q3/transfer_and_augmentation_models.joblib',
  'frozen_q1':AWORK/'q1/sensory_ensemble.joblib',
  'reference_script':HERE.parent/'budget_transfer/run_budget_transfer.py',
  'support_definitions':HERE.parent/'budget_transfer/selected_supports.csv',
  'old_diagnostics':HERE.parent/'budget_transfer/selected_diagnostics.csv',
  'old_profiles':HERE.parent/'budget_transfer/selected_profiles.csv'}
 hash_match={k:sha256(current_inputs[k])==expected for k,expected in protocol['input_sha256'].items()}
 documented_changes=audit_documented_metadata_changes(protocol,current_inputs,hash_match,features)
 scientific_match={k:unchanged or documented_changes.get(k,{}).get('passed',False) for k,unchanged in hash_match.items()}
 summary=dict(all_lp_witnesses_pass=bool(lp_audit.pass_audit.all()),lp_cases=len(lp_audit),feasible_witnesses=int(lp_audit.reported_feasible.sum()),
  infeasible_certificates=int((~lp_audit.reported_feasible).sum()),minimum_independent_separation_lower_bound=float(lp_audit.independent_separation_lower_bound.min()),
  all_alternative_lp_checks_agree=bool(ipm_audit.agrees_with_original.all()),alternative_lp_checks=len(ipm_audit),all_selected_candidates_pass=bool(ca.pass_audit.all()),selected_candidates=len(ca),
  exact_frozen_fields_in_serialized_profiles=bool(ca.frozen_raw_exact.all()),technical_internal_standard_exact=bool(ca.internal_standard_exact.all()),
  maximum_auxiliary_error=float(ca.auxiliary_max_error.max()),maximum_kcal_error=float(ca.kcal_error.max()),maximum_metric_error=float(ca.recomputed_metric_max_error.max()),
  nested_support_linear_feasibility_monotone=all(monotonic),old_input_files_unchanged=all(hash_match.values()),input_hash_checks=hash_match,
  documented_metadata_only_changes=documented_changes,input_scientific_content_checks=scientific_match,input_scientific_content_unchanged=all(scientific_match.values()),seconds=time.perf_counter()-t)
 (HERE/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 assert summary['all_lp_witnesses_pass'] and summary['all_alternative_lp_checks_agree'] and summary['all_selected_candidates_pass'] and summary['input_scientific_content_unchanged']

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
