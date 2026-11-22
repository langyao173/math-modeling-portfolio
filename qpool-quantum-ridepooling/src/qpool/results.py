"""Strict result parsing and separation of raw hardware output from classical repair."""
from pathlib import Path
import json
import numpy as np
from .qubo import PackingInstance,packing_qubo,read_matrix,decode_spins,repair


def read_solutions(path):
    path=Path(path)
    if path.suffix.lower()=='.json':
        obj=json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(obj,dict):
            if 'solutions' not in obj:raise ValueError('JSON object requires an explicit solutions array; no schema guessing.')
            obj=obj['solutions']
        a=np.asarray(obj)
    else:
        try:a=np.loadtxt(path,delimiter=',',ndmin=2)
        except ValueError as e:
            raise ValueError('Expected numeric CSV rows only. Preserve the original report; explicitly convert its solution field.') from e
    if a.ndim==1:a=a[None,:]
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Solutions must be a finite 2D numeric array.')
    return a


def inspect_solutions(meta_file,solutions,representation='ising',source='unverified_external'):
    meta_file=Path(meta_file);m=json.loads(meta_file.read_text());inst=PackingInstance.from_dict(m)
    Q=packing_qubo(inst,m['penalty_margin']);J=read_matrix(meta_file.parent/(m['name']+'_ising.csv'))
    n=len(inst.weights);rows=[]
    for k,s in enumerate(np.asarray(solutions)):
        if representation=='ising':x=decode_spins(s,n);H=float(-s@J@s)
        elif representation=='qubo':
            x=np.asarray(s)
            if x.shape!=(n,) or not np.isin(x,[0,1]).all():raise ValueError('Wrong bit count or non-binary result.')
            x=x.astype(int);sp=np.r_[2*x-1,1];H=float(-sp@J@sp)
        else:raise ValueError('Choose ising or qubo explicitly.')
        e=float(x@Q@x);ev=inst.evaluate(x)
        if not np.isclose(e,H/m['ising_scale']+m['qubo_bias']):raise ValueError('Energy identity failed.')
        rx,rinfo=repair(inst,x)
        row={'row':k,'bits':x.tolist(),'selected_variables_zero_based':np.flatnonzero(x).tolist(),
             'raw_reward':ev['reward'],'raw_feasible':ev['feasible'],'raw_conflicts':ev['n_conflicts'],
             'qubo_energy':e,'ising_energy':H,'classically_repaired_reward':rinfo['repaired']['reward'],
             'repaired_bits':rx.tolist()}
        if 'original_weights' in m:
            row['raw_original_utility']=float(np.asarray(m['original_weights'])@x)
            row['repaired_original_utility']=float(np.asarray(m['original_weights'])@rx)
        if 'lifting' in m:
            lift=m['lifting'];full=PackingInstance.from_dict(lift['full_instance'])
            fx=np.zeros(len(full.weights),int);fx[lift['fixed']]=1;fx[lift['kept']]=x
            fcheck=full.evaluate(fx)
            row['lifted_full_bits']=fx.tolist()
            row['lifted_raw_feasible']=fcheck['feasible']
            row['lifted_raw_original_utility']=float(full.weights@fx)
            seed=np.asarray(lift['seed'],int)
            row['seed_original_utility']=float(full.weights@seed)
            row['strictly_improves_seed_original_utility']=bool(fcheck['feasible'] and full.weights@fx>full.weights@seed+1e-10)
            # This describes the action rule; it does not overwrite the raw result.
            row['controller_would_keep_seed']=not row['strictly_improves_seed_original_utility']
            if ev['feasible'] and 'original_weights' in m:
                from .competition import polish_equal_quantized_reward
                core_original=PackingInstance(np.asarray(m['original_weights']),inst.edges,inst.labels)
                px,pinfo=polish_equal_quantized_reward(core_original,inst,x,.02)
                fp=np.zeros(len(full.weights),int);fp[lift['fixed']]=1;fp[lift['kept']]=px
                row['tie_polish']=pinfo
                row['tie_polished_full_original_utility']=float(full.weights@fp)
                row['tie_polished_bits']=px.tolist()
                row['tie_polished_full_feasible']=bool(full.evaluate(fp)['feasible'])
                row['controller_would_keep_seed_after_tie_polish']=not bool(full.evaluate(fp)['feasible'] and full.weights@fp>full.weights@seed+1e-10)

        rows.append(row)
    raw=[r for r in rows if r['raw_feasible']]
    return {'source_label':source,'physical_run_verified':False,
            'physical_verification_requirement':'Attach provider task report, task ID, model, timestamps and credit record.',
            'n_returned_rows':len(rows),'rows_are_not_assumed_independent_runs':True,
            'raw_feasible_rate':None if not rows else len(raw)/len(rows),
            'best_raw_feasible_reward':None if not raw else max(r['raw_reward'] for r in raw),
            'rows':rows}
