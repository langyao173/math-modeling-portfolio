"""Fair fixed-event generation-only timing; no sensory inference or fitting.

The two samplers target exactly the same fixed fitted P_style(X|B). Their
underlying acceptance probability is unchanged. Both rejection loops use
256-proposal batches and stop once >=100 successes have been accumulated.
The final batch is included in proposal/vector counts. Cholesky/Sigma
conditioning setup is separately recorded, including nonamortized cold time.
"""
import os,sys
sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time,json,argparse
import numpy as np,pandas as pd
from physical_generator import CEILINGS,feasibility,CeilingConditionedIntervalGaussian
from run_generation import PROJECT,BASE,atomic_to_csv,writej
from conditional_model import IntervalCopula,DERIVED,TECHNICAL
OUT=BASE/'results'
CEIL={k:v for k,v in CEILINGS.items() if k in ['DMS','X2.3.butanedione']}
SEEDS=[2026102501,2026102502,2026102503]

def full_rejection(model,features,style,n,rng,batch=256):
    accepted=[];proposed=0;got=0;p=len(features)
    while got<n:
        latent=model.mean_[style]+rng.standard_normal((batch,p))@model.chol_[style].T
        x=model.inverse_latent(latent)
        ok,_=feasibility(x,features,CEIL);xs=x[ok];accepted.append(xs);got+=len(xs);proposed+=batch
        if proposed>5000000:raise RuntimeError('Proposal budget exhausted')
    x=np.vstack(accepted)[:n]
    assert len(x)==n and feasibility(x,features,CEIL)[0].all()
    return x,proposed,got

def main():
    global OUT
    parser=argparse.ArgumentParser(description='Benchmark identical-event 225D rejection versus 2D marginal conditioning; no sensory inference. Uses MATHORCUP_PROJECT_DIR or ../project/MathorCup_A.')
    parser.add_argument('--out',type=Path,default=BASE/'results',help='Output directory within this generation directory')
    args=parser.parse_args();OUT=args.out.resolve()
    if not OUT.is_relative_to(BASE):raise ValueError('Output must stay under generation work directory')
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(PROJECT/'data/A/1.csv');features=[c for c in df.columns[3:] if c not in DERIVED+TECHNICAL]
    model=IntervalCopula(kind='interval_gaussian',kappa=2,nu=20).fit(df[features],df.tasting_category_fine)
    rows=[];outputs={}
    for style in ['Tripel','Kriek','Lambic']:
        # Warm both code paths with separate seeds; do not retain warm-up data.
        full_rejection(model,features,style,10,np.random.default_rng(117),batch=256)
        warm=CeilingConditionedIntervalGaussian(model,features,CEIL,styles=[style]);warm.sample(style,10,np.random.default_rng(117),block_size=256,return_rank=False)
        for k,seed in enumerate(SEEDS):
            order=['full225_rejection','marginal2_conditional'] if k%2==0 else ['marginal2_conditional','full225_rejection']
            for method in order:
                setup=0.
                if method=='marginal2_conditional':
                    t=time.perf_counter();sampler=CeilingConditionedIntervalGaussian(model,features,CEIL,styles=[style]);setup=time.perf_counter()-t
                t=time.perf_counter();rng=np.random.default_rng(seed)
                if method=='full225_rejection':
                    x,proposed,got=full_rejection(model,features,style,100,rng)
                    fullvectors=proposed;marginalvectors=0
                else:
                    x,z,rec=sampler.sample(style,100,rng,block_size=256,return_rank=False)
                    assert rec.complete;proposed=rec.proposed;got=round(rec.acceptance_estimate*rec.proposed)
                    fullvectors=100;marginalvectors=proposed
                elapsed=time.perf_counter()-t
                row=dict(style=style,seed=seed,method=method,target_n=100,accepted_output=len(x),event_pass_rate=float(feasibility(x,features,CEIL)[0].mean()),total_proposals=proposed,proposal_dimension=225 if method=='full225_rejection' else 2,full_225_vectors_generated=fullvectors,marginal_2_vectors_generated=marginalvectors,successes_in_all_final_batches=got,sampling_inverse_seconds=elapsed,extra_conditioning_setup_seconds=setup,cold_request_seconds=elapsed+setup)
                rows.append(row);outputs[f'{style}_{seed}_{method}']=x
                print(row,flush=True)
    data=pd.DataFrame(rows);data.to_csv(OUT/'sampling_efficiency_trials.csv',index=False)
    agg=data.groupby(['style','method']).agg(repetitions=('seed','count'),output_each=('accepted_output','min'),median_proposals=('total_proposals','median'),median_full225_vectors=('full_225_vectors_generated','median'),median_sampling_seconds=('sampling_inverse_seconds','median'),min_sampling_seconds=('sampling_inverse_seconds','min'),max_sampling_seconds=('sampling_inverse_seconds','max'),median_setup_seconds=('extra_conditioning_setup_seconds','median'),median_cold_seconds=('cold_request_seconds','median')).reset_index();agg.to_csv(OUT/'sampling_efficiency_summary.csv',index=False)
    writej(dict(seeds=SEEDS,n_per_trial=100,styles=['Tripel','Kriek','Lambic'],event=CEIL,batch_size=256,model='One shared IntervalCopula interval_gaussian kappa2 nu20 fit on all 250 supplied beers',timing='Only Gaussian sampling, constraint screening and empirical inverse. Both exclude model fitting, predictor inference, and rank-map forward transform. Additional Schur setup is reported separately and added to cold_request_seconds. Wall time on shared runtime; 3 repeats, no benchmark isolation.',interpretation='Marginal conditioning changes computation, not the event probability or target conditional law. Rejection counts vary with random streams, even for the same numeric seed. No rare-event probability improvement is claimed.'),OUT/'sampling_efficiency_protocol.json')
    # Buffered atomic compressed arrays, immediately read back.
    import io
    buffer=io.BytesIO();np.savez_compressed(buffer,**outputs);payload=buffer.getvalue();tmp=OUT/'sampling_efficiency_outputs.npz.atomic.tmp'
    with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,OUT/'sampling_efficiency_outputs.npz')
    with np.load(OUT/'sampling_efficiency_outputs.npz') as zz:
        assert set(zz.files)==set(outputs)
        for key in outputs:np.testing.assert_array_equal(zz[key],outputs[key])
    print(agg.to_string(index=False),flush=True)
if __name__=='__main__':main()
