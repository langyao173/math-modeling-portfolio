"""Read-only audit of saved generation results; no fitting or new sampling."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from physical_generator import feasibility
BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=BASE/'results',help='Existing generation result directory; read only')
    args=parser.parse_args();root=args.results.resolve()
    protocol=json.loads((root/'protocol.json').read_text());run=json.loads((root/'run_summary.json').read_text())
    assert run['complete'], 'Run is not marked complete'
    policies=protocol['policies'];counts={}
    for policy,ceilings in policies.items():
        frame=pd.read_csv(root/f'{policy}_conditioned_chemical_profiles.csv')
        valid,_=feasibility(frame[list(ceilings)].to_numpy(float),list(ceilings),ceilings)
        assert valid.all(), f'{policy} contains a ceiling violation'
        assert frame.synthetic_id.is_unique, f'{policy} has duplicate synthetic identifiers'
        assert len(frame)==run['actual_conditioned_counts'][policy], f'{policy} row count differs from saved run summary'
        counts[policy]=len(frame)
        if policy=='attachment_two':
            assert len(frame)==2200 and frame.tasting_category_fine.nunique()==22
            assert frame.tasting_category_fine.value_counts().eq(100).all()
    if run.get('cv_complete'):
        scores=pd.read_csv(root/'heldout_scores_by_beer.csv')
        primary=scores[(scores.policy=='attachment_two') & (scores.method=='conditional')]
        assert len(primary)==250 and primary.beer.nunique()==250
        selected=primary[primary.observed_passes_ceilings]
        assert len(selected)==29 and selected['style'].nunique()==14
        quality=pd.read_csv(root/'heldout_quality_summary.csv')
        q=quality[(quality.policy=='attachment_two')&(quality.method=='conditional')&(quality.evaluation_population=='ceiling_passing_validation')].iloc[0]
        np.testing.assert_allclose(selected.energy.mean(),q.energy,rtol=0.,atol=1e-14)
    benchmark=root/'sampling_efficiency_outputs.npz'
    benchmark_arrays=0
    if benchmark.is_file():
        frame=pd.read_csv(root/'attachment_two_conditioned_chemical_profiles.csv',nrows=1)
        omit={'synthetic_id','tasting_category_fine','kcalperc','esters_sum','aroma_hops_sum','acids_sum','sulfur_sum','X3.methyl.2.butenal'}
        features=[c for c in frame.columns if c not in omit]
        assert len(features)==225
        with np.load(benchmark) as arrays:
            benchmark_arrays=len(arrays.files);assert benchmark_arrays==18
            for key in arrays.files:
                x=arrays[key];assert x.shape==(100,225)
                assert feasibility(x,features,policies['attachment_two'])[0].all(), key
    print(json.dumps(dict(passed=True,candidate_counts=counts,cv_cache_checked=bool(run.get('cv_complete')),benchmark_arrays_checked=benchmark_arrays),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
