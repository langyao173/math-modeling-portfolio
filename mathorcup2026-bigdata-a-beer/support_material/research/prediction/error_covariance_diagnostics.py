"""Report residual and squared-loss covariance without fitting any model."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=ROOT/'fusion_replay_v1')
OUT=parser.parse_args().results
with np.load(OUT/'outer_predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
y=a['Y'];ea=a['raw231_y16_no_residual']-y;eb=a['raw231_y50_native']-y
rows=[]
for f in [None,0,1,2,3,4]:
    ix=np.ones(len(y),bool) if f is None else a['fold']==f
    va=y[ix].var(0);aa=ea[ix];bb=eb[ix]
    for j,t in enumerate(a['targets']):
        x,z=aa[:,j],bb[:,j];la,lb=x*x,z*z
        mse_a=float(la.mean());mse_b=float(lb.mean());cross=float((x*z).mean());dis=float(((x-z)**2).mean());half=float(np.mean(np.square((x+z)/2)))
        assert abs(half-(mse_a+mse_b+2*cross)/4)<1e-12
        rows.append({'fold':'pooled' if f is None else f,'target':t,'target_group':'core30' if j<30 else 'other20','n':int(ix.sum()),'truth_variance_evaluation_only':float(va[j]),'y16_bias':float(x.mean()),'native50_bias':float(z.mean()),'y16_mse':mse_a,'native50_mse':mse_b,'uncentered_error_cross_moment':cross,'centered_error_covariance':float(np.mean((x-x.mean())*(z-z.mean()))),'error_correlation':float(np.corrcoef(x,z)[0,1]),'squared_loss_covariance':float(np.mean((la-la.mean())*(lb-lb.mean()))),'squared_loss_correlation':float(np.corrcoef(la,lb)[0,1]),'squared_disagreement':dis,'half_blend_mse':half,'half_blend_mse_improvement_over_average':dis/4,'half_blend_r2_delta_vs_y16':(mse_a-half)/va[j],'half_blend_r2_delta_vs_native50':(mse_b-half)/va[j],'posthoc_per_target_optimal_y16_weight_DIAGNOSTIC_ONLY':float(np.clip((mse_b-cross)/dis,0,1)) if dis>1e-15 else .5})
frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_target_error_and_loss_covariance.csv',index=False)
print(frame.query('fold=="pooled"').groupby('target_group')[['error_correlation','squared_loss_correlation','half_blend_r2_delta_vs_y16','half_blend_r2_delta_vs_native50']].mean().to_string())
