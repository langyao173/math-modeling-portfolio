"""Audit generated profiles and produce exact, exportable scientific figures.

Figures display executed experimental results and never fabricated examples.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/mathorcup-q3-mpl')
import json,joblib
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transfer_profiles import OUT,DEFECTS,as_raw,enrich
from run_q3 import DEFAULT_DATA
from conditional_model import TECHNICAL

COLORS=['#70777E','#247B70','#C48B42','#3B7385','#8F9CA3']

def save(fig,name):
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
 fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight')
 plt.close(fig)

def main():
 df=pd.read_csv(DEFAULT_DATA/'1.csv');cal=pd.read_excel(DEFAULT_DATA/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer');bundle=joblib.load(OUT/'transfer_and_augmentation_models.joblib');control=bundle['model'];features=bundle['features']
 gen=pd.read_csv(OUT/'revised_generated_chemical_profiles.csv');proto=pd.read_csv(OUT/'revised_style_prototypes.csv')
 gen=enrich(gen,df,features,control.map_,bundle['derived_predictors']);proto=enrich(proto,df,features,control.map_,bundle['derived_predictors'])
 gen.to_csv(OUT/'revised_generated_chemical_profiles_complete.csv',index=False);proto.to_csv(OUT/'revised_style_prototypes_complete.csv',index=False)
 zg=control.map_.transform(gen[features]);z=control.z_;styles=df.tasting_category_fine.to_numpy(str)
 global_nn=cdist(zg,z).min(axis=1)/np.sqrt(len(features));audit=gen[['synthetic_id','tasting_category_fine']].copy();audit['nearest_training_distance']=global_nn
 for s in control.styles_:
  ix=np.flatnonzero(gen.tasting_category_fine.to_numpy(str)==s);ir=np.flatnonzero(styles==s)
  nn=cdist(zg[ix],z[ir]).min(axis=1)/np.sqrt(len(features))
  realdist=cdist(z[ir],z[ir])/np.sqrt(len(features));np.fill_diagonal(realdist,np.inf)
  threshold=np.quantile(realdist.min(axis=1),.95)
  audit.loc[ix,'same_style_nearest_distance']=nn;audit.loc[ix,'real_style_loo_nn_q95']=threshold;audit.loc[ix,'above_real_nn_q95']=nn>threshold
 audit.to_csv(OUT/'generated_profile_diagnostics.csv',index=False)
 defectsummary=[]
 for s in control.styles_:
  for col,info in DEFECTS.items():
   real=as_raw(df.loc[df.tasting_category_fine==s,col].to_numpy(float),info);new=as_raw(gen.loc[gen.tasting_category_fine==s,col].to_numpy(float),info);t=info['threshold']
   source_rows=df.loc[df.tasting_category_fine==s];source_raw=cal.loc[source_rows.beer,col].to_numpy(float)
   replacement=(info['transform']=='log10') & ((source_raw<=0)|~np.isfinite(source_raw)) & np.isfinite(real)
   defectsummary.append(dict(style=s,chemical=col,unit=info['unit'],threshold=t,n_input_present=int(np.isfinite(real).sum()),n_zero_or_nonpositive_replacements=int(np.sum(replacement)),n_missing=int(np.sum(~np.isfinite(real))),reference_exceedance_rate_in_input_representation=float(np.mean(real[np.isfinite(real)]>t)),generated_exceedance_rate=float(np.mean(new>t)),reference_median_in_input_representation=float(np.nanmedian(real)),generated_median=float(np.median(new)),generated_q10=float(np.quantile(new,.1)),generated_q90=float(np.quantile(new,.9))))
 pd.DataFrame(defectsummary).to_csv(OUT/'generated_defect_comparison.csv',index=False)
 requested=['Tripel','Kriek','Lambic'];representatives=[]
 for s in requested:
  ids=np.flatnonzero(gen.tasting_category_fine.to_numpy(str)==s);center=np.median(zg[ids],axis=0)
  chosen=ids[np.linalg.norm(zg[ids]-center,axis=1).argmin()];representatives.append(gen.iloc[chosen])
 pd.DataFrame(representatives).to_csv(OUT/'requested_style_representatives.csv',index=False)
 # Paired difference CI respects the same held-out beer in each outer fold.
 bs=pd.read_csv(OUT/'extension_outer_scores_by_beer.csv');selected=bs[bs.selected].set_index('beer').energy;ci=[]
 rng=np.random.default_rng(20261023)
 for name in bs.model.unique():
  base=bs[bs.model==name].set_index('beer').energy.loc[selected.index]
  d=(base-selected).to_numpy();boot=d[rng.integers(0,len(d),(3000,len(d)))].mean(axis=1)
  ci.append(dict(reference=name,n=len(d),selected_process_energy_reduction=d.mean(),ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975)))
 pd.DataFrame(ci).to_csv(OUT/'extension_paired_energy_intervals.csv',index=False)
 # Sensitivity of the final engineering selection rule; no new outer evaluation.
 inner=pd.read_csv(OUT/'extension_inner_selection.csv');selection_sensitivity=[]
 for auc_weight in [.05,.15,.30]:
  for neighbor_weight in [0.,.05,.10]:
   scores=inner.assign(sensitivity_score=inner.energy+auc_weight*np.maximum(0,inner.distinguishability-.65)+neighbor_weight*np.maximum(0,.5-inner.nn_ratio)).groupby('model').sensitivity_score.mean()
   selected=scores.idxmin()
   selection_sensitivity.append(dict(auc_weight=auc_weight,neighbor_weight=neighbor_weight,selected_model=selected,mean_selection_score=float(scores[selected])))
 pd.DataFrame(selection_sensitivity).to_csv(OUT/'selection_weight_sensitivity.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#273239','text.color':'#273239','axes.edgecolor':'#69757B','xtick.color':'#47545B','ytick.color':'#47545B','svg.fonttype':'none'})
 # Generator quality: three metrics answer three different questions.
 d=pd.read_csv(OUT/'extension_outer_comparison.csv');names=['smooth_pooled','interval_gaussian','interval_kernel_h03','interval_kernel_h06','interval_kernel_h09'];labels=['Smooth copula','Interval copula','Kernel h=0.3','Kernel h=0.6','Kernel h=0.9']
 fig,axs=plt.subplots(1,3,figsize=(11.2,3.5),sharey=True,gridspec_kw={'wspace':.32})
 for ax,col,title in zip(axs,['energy','c2st_auc','nn_ratio'],['A   Predictive distribution','B   Distinguishability','C   Novelty']):
  for j,name in enumerate(names):
   vals=d[d.model==name][col].to_numpy();ax.plot(vals,np.full(len(vals),j),'o',ms=4,color=COLORS[j],alpha=.5);ax.plot(np.mean(vals),j,'|',ms=13,mew=2.5,color=COLORS[j])
  ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.grid(axis='x',color='#E7E9E9',lw=.7);ax.set_axisbelow(True);ax.set_title(title,loc='left',fontsize=10,pad=12)
 axs[0].set_xlabel('Energy score (lower is better)');axs[1].set_xlabel('Real / synthetic AUC');axs[1].axvline(.5,color='#8A9398',ls='--',lw=.8);axs[1].set_xlim(.45,1.01);axs[2].set_xlabel('Nearest-neighbor distance ratio\n(synthetic / held-out real)');axs[2].axvline(1,color='#8A9398',ls='--',lw=.8)
 fig.text(.01,-.10,'Points: three outer validation folds. Thick bars: fold means. Every transform is fitted within the training fold.',fontsize=8,color='#5F6C73')
 save(fig,'q3_generator_validation')
 # Transfer controls: improvement and cost shown next to safety counts.
 t=pd.read_csv(OUT/'transfer_comparison_metrics.csv');methods=['unconstrained','coordinate_clipping','joint_qp','strict_concentration_qp'];mlabs=['Unconstrained','Coordinate clipping','Joint constraints','Strict concentrations'];fig,axs=plt.subplots(1,3,figsize=(11.2,3.6),sharey=True,gridspec_kw={'wspace':.35})
 for j,m in enumerate(methods):
  v=t[t.method==m];color=['#879198','#B29468','#247B70','#3B7385'][j]
  axs[0].scatter(100*v.target_distance_reduction,np.full(len(v),j),s=15,color=color,alpha=.5);axs[0].plot(100*v.target_distance_reduction.mean(),j,'|',mew=2.5,ms=13,color=color)
  axs[1].scatter(v.intervention_rms,np.full(len(v),j),s=15,color=color,alpha=.5);axs[1].plot(v.intervention_rms.mean(),j,'|',mew=2.5,ms=13,color=color)
  vals=[sum(v.max_chemical_violation>1e-8),sum(v.stale_proxy_violation>1e-8)]
  axs[2].barh(j-.13,vals[0],height=.22,color='#AC614E',label='Chemical exceedance' if j==0 else None)
  axs[2].barh(j+.13,vals[1],height=.22,color='#6F8695',label='Stale-hops proxy' if j==0 else None)
  for off,n in zip([-.13,.13],vals):axs[2].text(n+.15,j+off,str(n),va='center',fontsize=8)
 for ax in axs:ax.set_yticks(range(4),mlabs);ax.invert_yaxis();ax.grid(axis='x',color='#E7E9E9',lw=.7);ax.set_axisbelow(True)
 axs[0].set_title('A   Approach to target style',loc='left',fontsize=10,pad=12);axs[0].set_xlabel('Reduction in target distance (%)')
 axs[1].set_title('B   Chemical intervention',loc='left',fontsize=10,pad=12);axs[1].set_xlabel('RMS change in rank space')
 axs[2].set_title('C   Violations among 11 source beers',loc='left',fontsize=10,pad=12);axs[2].set_xlabel('Number of violations');axs[2].set_xlim(0,12);axs[2].legend(frameon=False,loc='lower right',fontsize=7)
 fig.text(.01,-.055,'Preservation weight = 0.2. Stale-hops constraints refer to a fitted proxy; true sensory quality needs prospective tasting.',fontsize=8,color='#5F6C73')
 save(fig,'q3_transfer_controls')
 # Main tradeoff is an explicit, reproducible intervention-strength choice.
 f=pd.read_csv(OUT/'transfer_tradeoff.csv');fig,ax=plt.subplots(figsize=(5.6,3.5))
 summary=f.groupby('lambda_preservation').agg(gain=('target_distance_reduction','mean'),change=('intervention_rms','mean'))
 ax.plot(summary.change,100*summary.gain,'-',color='#A8B7B6',lw=1.2)
 for w,r in summary.iterrows():
  ax.scatter(r.change,100*r.gain,s=55,color='#247B70' if w==.2 else '#8A9C9C',zorder=3);ax.annotate(f'lambda = {w:g}',(r.change,100*r.gain),xytext=(6,-12 if w==.2 else 7),textcoords='offset points',fontsize=8)
 ax.set_xlabel('Mean RMS chemical change');ax.set_ylabel('Mean target-distance reduction (%)');ax.grid(color='#E7E9E9',lw=.7);ax.set_axisbelow(True);ax.set_title('Preservation strength controls the intervention cost',loc='left',fontsize=10,pad=12)
 save(fig,'q3_transfer_tradeoff')
 # Original and generated style fingerprints in a common empirical scale.
 compounds=['ethanol..v.v.','ethyl_acetate','isopentyl_acetate','phenethyl_alcohol','lactic_acid.mg.L.','acetic_acid.g.L.','bitterness','color.EBC.','DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']
 labels=['Ethanol','Ethyl acetate','Isoamyl acetate','Phenethyl alcohol','Lactic acid','Acetic acid','Iso-alpha acids','Color','DMS','Diacetyl','4-vinylguaiacol','trans-2-nonenal'];arr=np.zeros((len(compounds),3));finger=[]
 for j,c in enumerate(compounds):
  v=np.sort(df[c].dropna());
  for k,s in enumerate(requested):
   q=np.quantile(gen.loc[gen.tasting_category_fine==s,c],[.1,.5,.9]);pct=np.searchsorted(v,q[1])/len(v);arr[j,k]=pct
   finger.append(dict(style=s,chemical=c,q10=q[0],median=q[1],q90=q[2],median_reference_percentile=pct,scale='original 1.csv field scale'))
 pd.DataFrame(finger).to_csv(OUT/'requested_style_fingerprints.csv',index=False)
 fig,ax=plt.subplots(figsize=(5.2,5.3));im=ax.imshow(arr,cmap='BrBG',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(3),requested);ax.set_yticks(range(len(labels)),labels);ax.tick_params(length=0)
 for j in range(len(compounds)):
  for k in range(3):ax.text(k,j,f'{100*arr[j,k]:.0f}',ha='center',va='center',fontsize=8,color='white' if abs(arr[j,k]-.5)>.32 else '#223A3D')
 ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(labels),1),minor=True);ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',length=0);cb=fig.colorbar(im,ax=ax,shrink=.65,pad=.08);cb.set_label('Median percentile among observed beers');ax.set_title('Generated chemical fingerprints',loc='left',fontsize=11,pad=14)
 save(fig,'q3_style_fingerprints')
 meta=dict(n_generated=len(gen),n_styles=int(gen.tasting_category_fine.nunique()),direct_model_features=len(features),technical_fields=TECHNICAL,exact_copy_rate=float(np.mean(global_nn<1e-8)),nearest_neighbor_median=float(np.median(global_nn)),n_beyond_style_nn_q95=int(audit.above_real_nn_q95.sum()),interpretation='Support and similarity diagnostics only, not physical brewing or sensory validation.',representatives={r['tasting_category_fine']:r['synthetic_id'] for r in representatives})
 (OUT/'final_generation_audit.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
 print(json.dumps(meta,ensure_ascii=False))

if __name__=='__main__':main()
