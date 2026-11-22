#!/usr/bin/env python3
"""Independent measurement-status, support, and cross-fit proxy diagnostics.

No candidates are selected or modified by this diagnostic. Model agreement is
not a human tasting result, and source-held-out models are not new external data.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score

OUT=Path(__file__).parent
Q1=OUT.parent/'q1';Q3=OUT.parent/'q3'
sys.path.insert(0,str(Q1));sys.path.insert(0,str(Q3))
from sensory_model import predict_bundle
from transfer_profiles import DEFECTS,as_raw
DATA=Path(os.environ.get('MATHORCUP_DATA_DIR','/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件'))


def kcal_from_fields(d):
    raw={c:10.**d[c].to_numpy(float) for c in ['ethanol..v.v.','GFS','unfermentables','protein.g.L.','lactic_acid.mg.L.']}
    return np.log10(5.523*raw['ethanol..v.v.']+.4*(raw['GFS']+raw['unfermentables']+raw['protein.g.L.'])+.0003*raw['lactic_acid.mg.L.'])


def main():
    source=pd.read_csv(DATA/'1.csv').set_index('beer')
    original_cal=pd.read_excel(DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer')
    final=pd.read_csv(Q3/'transfer_final_profiles.csv').set_index('beer')
    published=pd.read_csv(Q3/'transfer_final_chemical_certificate.csv').set_index(['beer','chemical'])
    fold=pd.read_csv(Q1/'fold_assignments.csv').set_index('beer')['fold']
    full=joblib.load(Q1/'sensory_ensemble.joblib')
    src=source.loc[final.index].copy()
    source_full=predict_bundle(full,src);new_full=predict_bundle(full,final)
    full_old=source_full.stale_hops.to_numpy();full_new=new_full.stale_hops.to_numpy()
    records=[];status_rows=[]
    for k,beer in enumerate(final.index):
        f=int(fold.loc[beer]);bundle=joblib.load(Q1/f'fold_{f}_models.joblib')
        before=float(predict_bundle(bundle,source.loc[[beer]])['stale_hops'].iloc[0])
        after=float(predict_bundle(bundle,final.loc[[beer]])['stale_hops'].iloc[0])
        records.append(dict(beer=int(beer),held_out_fold=f,method=final.loc[beer,'method'],
            source_stale_oof_model=before,transfer_stale_oof_model=after,stale_delta_oof_model=after-before,
            stale_increase_oof_model=bool(after>before+1e-8),source_stale_full_model=float(full_old[k]),
            transfer_stale_full_model=float(full_new[k]),stale_delta_full_model=float(full_new[k]-full_old[k])))
        for c,info in DEFECTS.items():
            input0=source.loc[beer,c];reported=original_cal.loc[beer,c]
            rep=as_raw(input0,info);out=as_raw(final.loc[beer,c],info)
            if not np.isfinite(input0):status='missing'
            elif info['transform']=='log10' and (not np.isfinite(reported) or reported<=0):status='zero_or_nonpositive_replacement'
            else:status='positive_reported_measurement'
            pub=published.loc[(beer,c)]
            assert pub.source_status==status
            assert (pd.isna(pub.source_concentration) if status!='positive_reported_measurement' else np.isclose(pub.source_concentration,rep))
            status_rows.append(dict(beer=int(beer),chemical=c,source_status=status,source_model_representative=float(rep),
                transferred_concentration=float(out),threshold=info['threshold'],
                representation_increase=float(out-rep) if np.isfinite(rep) else None,
                measured_concentration_increase=float(out-rep) if status=='positive_reported_measurement' else None,
                output_exceedance=float(max(out/info['threshold']-1.,0.))))
    cross=pd.DataFrame(records)
    cross.to_csv(OUT/'transfer_oof_proxy_diagnostic.csv',index=False)
    status=pd.DataFrame(status_rows)
    status.to_csv(OUT/'transfer_independent_chemical_checks.csv',index=False)
    pack=joblib.load(Q3/'transfer_and_augmentation_models.joblib')
    control=pack['model'];fields=pack['features'];mp=control.map_
    A=final[fields].to_numpy(float)
    low=np.array([v[0][0] for v in mp.knots_]);high=np.array([v[0][-1] for v in mp.knots_])
    lowviol=np.maximum(low-A,0.);highviol=np.maximum(A-high,0.)
    Z=mp.transform(A);Z0=mp.transform(src[fields])
    linear_delta=pack['stale_proxy'].predict(Z)-pack['stale_proxy'].predict(Z0)
    derived=[]
    for c in ['esters_sum','aroma_hops_sum','acids_sum','sulfur_sum']:
        lo=float(source[c].min());hi=float(source[c].max());v=final[c].to_numpy(float)
        derived.append(dict(field=c,min=float(v.min()),max=float(v.max()),observed_min=lo,observed_max=hi,
                            n_below=int((v<lo-1e-9).sum()),n_above=int((v>hi+1e-9).sum())))
    gen=pd.read_csv(Q3/'revised_generated_chemical_profiles_complete.csv')
    # Point mass generator is checked against its imputed empirical support.
    ig=joblib.load(Q3/'revised_conditional_generator.joblib')
    if isinstance(ig,dict):ig=ig['model']
    support_failures=[]
    for j,c in enumerate(fields):
        values=np.sort(np.unique(ig.x_[:,j]));v=gen[c].to_numpy(float)
        ix=np.clip(np.searchsorted(values,v),0,len(values)-1)
        distance=np.minimum(np.abs(v-values[ix]),np.abs(v-values[np.maximum(ix-1,0)]))
        if np.any(distance>1e-8):support_failures.append(c)
    main_oof=pd.read_csv(Q1/'oof_predictions_standardized.csv').set_index('beer')
    sensory=pd.read_csv(DATA/'4.csv').set_index('beer')
    src_pred_error=float(np.max(np.abs(cross.source_stale_oof_model.to_numpy()-main_oof.loc[cross.beer,'stale_hops'].to_numpy())))
    assert src_pred_error<1e-8
    result=dict(n_sources=len(final),n_features=len(fields),source_status_counts=status.source_status.value_counts().to_dict(),
        max_measured_concentration_increase=float(status.measured_concentration_increase.max()),
        max_representative_increase=float(status.representation_increase.max()),
        n_missing_output_above_threshold=int(((status.source_status=='missing')&(status.output_exceedance>1e-8)).sum()),
        n_floor_output_above_threshold=int(((status.source_status=='zero_or_nonpositive_replacement')&(status.output_exceedance>1e-8)).sum()),
        global_marginal_support_max_violation=float(max(lowviol.max(),highviol.max())),
        transfer_kcal_identity_max_error=float(np.max(np.abs(final.kcalperc.to_numpy()-kcal_from_fields(final)))),
        generated_kcal_identity_max_error=float(np.max(np.abs(gen.kcalperc.to_numpy()-kcal_from_fields(gen)))),
        generated_empirical_coordinate_support_failures=support_failures,
        derived_index_support=derived,
        full_model_stale_max_delta=float(np.max(full_new-full_old)),linear_stale_max_delta=float(linear_delta.max()),
        crossfit_source_prediction_reproduction_max_error=src_pred_error,
        crossfit_stale_n_increased=int(cross.stale_increase_oof_model.sum()),
        crossfit_stale_increased_beer=cross.loc[cross.stale_increase_oof_model,'beer'].astype(int).tolist(),
        crossfit_stale_delta_mean=float(cross.stale_delta_oof_model.mean()),
        crossfit_stale_delta_min=float(cross.stale_delta_oof_model.min()),crossfit_stale_delta_max=float(cross.stale_delta_oof_model.max()),
        q1_stale_oof_r2=float(r2_score(sensory.loc[main_oof.index,'stale_hops'],main_oof.stale_hops)),
        interpretation='Independent numerical audit and source-held-out proxy sensitivity only; final candidates were designed with full-data models, so this is not an untouched external validation or a human sensory certificate.')
    (OUT/'final_transfer_independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
