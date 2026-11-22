"""Post-hoc joint filter of already generated and checked paths; no new search.

Added after seeing that a Tripel argmax can be achieved while moving farther
from the Tripel centroid. The classifier-only files remain unchanged.
"""
from pathlib import Path
import json,hashlib,os
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;AWORK=HERE.parents[1]
THRESHOLD=1e-8

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 paths=['candidate_checks.csv','selected_diagnostics.csv','selected_profiles.csv','selected_sensory_predictions.csv','audit_selected_candidates.csv','support_definitions.csv']
 hashes={x:sha(HERE/x) for x in paths}
 protocol=dict(new_exploratory_posthoc_filter=True,reason='Observed fixed-classifier membership can occur despite increasing target distance.',
  added_after_inspecting_classifier_only_results=True,target_distance_reduction_strictly_greater_than=THRESHOLD,
  base_acceptance='four original chemical bounds, rank box, unchanged direct fields/internal standard, original linear and frozen Q1 stale nonincrease, and Tripel logit halfspaces',
  pool='all already generated and checked candidate_checks.csv rows with accepted=True; no retraining, optimization or new path search',
  nested_budget='all existing forward supports at or below budget, with all_225 included at 225',
  selection='minimum unchanged quadratic objective; then raw field count, margin, alpha, stable order',
  fallback='original source with distance reduction=0 if no joint candidate exists',
  reporting='both all-11 mean with fallbacks and mean conditional on joint acceptance',source_sha256=hashes)
 (HERE/'joint_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
 c=pd.read_csv(HERE/'candidate_checks.csv',float_precision='round_trip');d=pd.read_csv(HERE/'selected_diagnostics.csv',float_precision='round_trip')
 profiles=pd.read_csv(HERE/'selected_profiles.csv',float_precision='round_trip');pred=pd.read_csv(HERE/'selected_sensory_predictions.csv',float_precision='round_trip')
 audit=pd.read_csv(HERE/'audit_selected_candidates.csv');supports=pd.read_csv(HERE/'support_definitions.csv')
 data=Path(os.environ.get('MATHORCUP_DATA_DIR',AWORK.parent/'data/A')).expanduser().resolve()
 df=pd.read_csv(data/'1.csv')
 joint=c.accepted & (c.target_distance_reduction>THRESHOLD)
 c[['candidate_index','beer','method','allowed_coordinates','accepted','target_distance_reduction']].assign(joint_accepted=joint).to_csv(HERE/'joint_path_screen.csv',index=False)
 mapping={int(row.candidate_index):i for i,row in d.iterrows() if row.accepted}
 sources={int(beer):int(g.sort_values('allowed_coordinates').index[0]) for beer,g in d.groupby('beer')}
 outdiag=[];outprofile=[];outpred=[];outaudit=[]
 for i,case in supports.iterrows():
  beer=int(case.beer);k=int(case.allowed_coordinates)
  eligible=c[(c.beer==beer)&(c.allowed_coordinates<=k)&joint]
  if len(eligible):
   best=eligible.sort_values(['quadratic_objective','raw_direct_fields_changed','margin','alpha'],kind='stable').iloc[0]
   index=int(best.candidate_index)
   # All selected profiles happen to be among the already persisted originals.
   assert index in mapping,'Existing final artifacts do not contain a needed candidate; do not rerun search silently.'
   old_i=mapping[index];rec=d.iloc[old_i].to_dict();row=profiles.iloc[old_i].to_dict();prediction=pred.iloc[old_i].to_dict()
   assert audit.iloc[old_i].pass_audit
   rec.update(method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),joint_accepted=True,
      selection_origin_method=best.method,source_fallback=False)
   row.update(method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),selection_origin_method=best.method,source_fallback=False,joint_accepted=True)
   prediction.update(method=case.method)
   proof=dict(beer=beer,method=case.method,allowed_coordinates=k,candidate_index=index,eligible_candidates=len(eligible),
      chosen_objective=float(best.quadratic_objective),minimum_eligible_objective=float(eligible.quadratic_objective.min()),
      minimum_objective_match=bool(abs(rec['quadratic_objective']-eligible.quadratic_objective.min())<1e-10),reused_previous_audited_candidate=True,source_fallback=False)
  else:
   old_i=sources[beer];rec=d.iloc[old_i].to_dict();row=profiles.iloc[old_i].to_dict();prediction=pred.iloc[old_i].to_dict()
   assert rec['source_fallback'] and not rec['accepted'] and audit.iloc[old_i].pass_audit
   rec.update(method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),joint_accepted=False)
   row.update(method=case.method,allowed_coordinates=k,selection_case_id=int(case.case_id),joint_accepted=False)
   prediction.update(method=case.method)
   proof=dict(beer=beer,method=case.method,allowed_coordinates=k,candidate_index=-1,eligible_candidates=0,chosen_objective=rec['quadratic_objective'],minimum_eligible_objective=np.nan,
      minimum_objective_match=True,reused_previous_audited_candidate=True,source_fallback=True)
  rec['selection_policy']='posthoc_joint_Tripel_and_positive_distance';row['selection_policy']=rec['selection_policy']
  outdiag.append(rec);outprofile.append(row);outpred.append(prediction);outaudit.append(proof)
 j=pd.DataFrame(outdiag);jp=pd.DataFrame(outprofile);pr=pd.DataFrame(outpred);au=pd.DataFrame(outaudit)
 j.to_csv(HERE/'joint_selected_diagnostics.csv',index=False);jp.to_csv(HERE/'joint_selected_profiles.csv',index=False);pr.to_csv(HERE/'joint_selected_sensory_predictions.csv',index=False);au.to_csv(HERE/'joint_selection_audit.csv',index=False)
 class_budget=pd.read_csv(HERE/'budget_summary.csv');rows=[]
 for k,g in j.groupby('allowed_coordinates',sort=True):
  valid=g[g.joint_accepted];old=class_budget[class_budget.budget==k].iloc[0]
  rows.append(dict(budget=int(k),n_sources=len(g),joint_accepted=int(g.joint_accepted.sum()),source_fallbacks=int(g.source_fallback.sum()),
    all_11_mean_distance_reduction=float(g.target_distance_reduction.mean()),accepted_conditional_mean_distance_reduction=float(valid.target_distance_reduction.mean()) if len(valid) else None,
    all_11_mean_intervention_rms=float(g.intervention_rms.mean()),accepted_conditional_mean_intervention_rms=float(valid.intervention_rms.mean()) if len(valid) else None,
    accepted_min_distance_reduction=float(valid.target_distance_reduction.min()) if len(valid) else None,
    accepted_mean_target_probability=float(valid.target_probability.mean()) if len(valid) else None,
    accepted_max_stale_delta=float(valid.ensemble_stale_delta.max()) if len(valid) else None,
    class_only_accepted=int(old.complete_accepted),class_only_all_11_mean_distance_reduction=float(old.all_source_mean_distance_reduction),
    linear_feasible_m0=int(old.linear_feasible_m0),old_distance_objective_tripel=int(old.old_tripel_classified),old_distance_objective_mean_distance_reduction=float(old.old_mean_distance_reduction)))
 summary=pd.DataFrame(rows);summary.to_csv(HERE/'joint_budget_summary.csv',index=False)
 first=pd.read_csv(HERE/'first_tested_budgets.csv');newfirst=[]
 for beer,g in j.groupby('beer',sort=True):
  valid=g[g.joint_accepted].sort_values('allowed_coordinates');prev=first[first.beer==beer].iloc[0].to_dict()
  prev['first_tested_joint_budget']=int(valid.allowed_coordinates.iloc[0]) if len(valid) else None
  if len(valid):
   best=valid.iloc[0];prev.update(joint_first_distance_reduction=float(best.target_distance_reduction),joint_first_actual_raw_fields=int(best.raw_direct_fields_changed),
    joint_first_intervention_rms=float(best.intervention_rms),joint_first_target_probability=float(best.target_probability),joint_first_stale_delta=float(best.ensemble_stale_delta))
  newfirst.append(prev)
 firstjoint=pd.DataFrame(newfirst);firstjoint.to_csv(HERE/'joint_first_tested_budgets.csv',index=False)
 keys={(int(r.beer),int(r.first_tested_joint_budget)) for r in firstjoint.itertuples() if pd.notna(r.first_tested_joint_budget)}
 chosen=j.apply(lambda r:(int(r.beer),int(r.allowed_coordinates)) in keys,axis=1)
 j.loc[chosen].to_csv(HERE/'joint_first_accepted_diagnostics.csv',index=False);jp.loc[chosen].to_csv(HERE/'joint_first_accepted_profiles.csv',index=False)
 baseline=df.set_index('beer');direct_features=json.loads((HERE/'protocol.json').read_text())['rank_dimensions']
 # Carry chemical certificates from the exact reused profiles; fallback uses source.
 cert=pd.read_csv(HERE/'chemical_certificate.csv');certrows=[]
 for i,rec in j.iterrows():
  old_i=mapping[int(rec.candidate_index)] if rec.joint_accepted else sources[int(rec.beer)]
  origin=d.iloc[old_i]
  sub=cert[(cert.beer==origin.beer)&cert.method.eq(origin.method)].copy();sub['method']=rec.method;sub['joint_accepted']=rec.joint_accepted
  certrows.append(sub)
 pd.concat(certrows,ignore_index=True).to_csv(HERE/'joint_chemical_certificate.csv',index=False)
 after_hashes={x:sha(HERE/x) for x in paths};assert hashes==after_hashes
 status=dict(posthoc=True,no_new_training_or_search=True,n_original_candidate_paths=len(c),n_original_accepted_paths=int(c.accepted.sum()),n_joint_accepted_paths=int(joint.sum()),
   n_budget_outputs=len(j),n_joint_accepted_budget_outputs=int(j.joint_accepted.sum()),all_sources_have_joint_candidate=bool(firstjoint.first_tested_joint_budget.notna().all()),
   all_selected_candidates_previously_audited=True,all_selection_minima_match=bool(au.minimum_objective_match.all()),class_only_files_unchanged=hashes==after_hashes,
   first_tested_budget_with_all_11=int(summary.loc[summary.joint_accepted==11,'budget'].min()))
 (HERE/'joint_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
 print(summary.to_string(index=False));print(firstjoint.to_string(index=False));print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__':main()
