"""Six predetermined scenario/scale checks after successful portal calibration.

The largest chronological candidate snapshot is selected in each scenario,
without reference to solver wins or quantum results. These are development
scale checks, not a randomly sampled test leaderboard.
"""
from pathlib import Path
import sys,json,csv,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.qubo import PackingInstance,write_matrix
from qpool.competition import prepare_hardware_plan
from qpool.solvers import solve_greedy,solve_local,solve_milp,solve_sa

root=Path('results/stage2/contest_replay');out=Path('upload/03_application_batch');out.mkdir(parents=True,exist_ok=True)
results=Path('results/stage2/application_batch');results.mkdir(parents=True,exist_ok=True)
rows=[];bench=[]
for index,(fleet,epoch) in enumerate([(16,60),(32,60),(48,60),(16,300),(32,300),(48,300)],start=1):
 source=root/f'f{fleet}_e{epoch}_s11_myopic_mip'
 snaps=json.loads((source/'snapshots.json').read_text())
 snap=max(snaps,key=lambda d:(len(d['weights']),-d['epoch']))
 full=PackingInstance.from_dict(snap);seed,si=solve_local(full,.02)
 p,other=prepare_hardware_plan(full,seed,max_variables=96,min_resolution=1.)
 if p is None:raise RuntimeError('This predefined case needs no hardware: '+str(other))
 name=f'QP_A{index:02d}';meta=p.meta.copy()
 meta.update(p.quantized.to_dict());meta.update({'name':name,'n_binary':len(p.kept),
   'purpose':'competition_application_scale_check','source_replay':str(source),'source_epoch':snap['epoch'],
   'fleet':fleet,'epoch_seconds':epoch,'selection_rule':'largest candidate count in each predefined scenario; earliest tie',
   'source_interpretation':'external Sioux Falls example; unchanged 240-request prefix; independent replay',
   'original_weights':p.core.weights.tolist(),'utility_quantization_resolution':p.meta['resolution'],
   'portal_schema_evidence':'numeric symmetric integer CSV accepted in QP_CHECK06_01; this file awaits portal validation',
   'full_day_quantum_replay':False,
   'lifting':{'kept':p.kept,'fixed':p.fixed,'seed':p.seed.tolist(),'full_instance':full.to_dict()},
   'candidate_routes':[snap['candidate_routes'][i] for i in p.kept]})
 x,info=solve_milp(p.quantized,20.)
 if not info['optimal']:raise RuntimeError('The calibration batch needs known reference optima.')
 spins=np.r_[2*x-1,1]
 original_x,original_info=solve_milp(full,20)
 core_orig,core_orig_info=solve_milp(p.core,20)
 meta.update({'exact_quantized_optimum':float(p.quantized.weights@x),'exact_quantized_bits':x.tolist(),
   'exact_ising_minimum':float(-spins@p.J@spins),
   'full_original_reference_utility':float(full.weights@original_x),'full_reference_proven_optimal':original_info['optimal'],
   'conditional_original_reference_utility':float(p.core.weights@core_orig)+meta['fixed_original_utility'],
   'quantized_optimizer_original_total_utility':float(full.weights@p.lift(x)),
   'seed_original_total_utility':float(full.weights@p.seed)})
 write_matrix(out/(name+'_ising.csv'),p.J)
 # Only Ising inputs live in the upload folder; QUBO is kept with supporting records.
 write_matrix(results/(name+'_qubo.csv'),p.Q)
 (out/(name+'.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False))
 (results/(name+'_snapshot.json')).write_text(json.dumps(snap,indent=2))
 row={'name':name,'fleet':fleet,'epoch_seconds':epoch,'decision_time':snap['epoch'],
      'full_candidates':len(full.weights),'upload_variables':len(p.kept),'ising_spins':len(p.J),
      'conflict_edges':len(p.core.edges),'scope':p.meta['scope'],'precision':p.meta['resolution'],
      'minimum_ising':int(p.J.min()),'maximum_ising':int(p.J.max()),
      'known_minimum_H':meta['exact_ising_minimum'],'known_quantized_reward':meta['exact_quantized_optimum'],
      'full_original_optimum':meta['full_original_reference_utility'],
      'conditional_original_optimum':meta['conditional_original_reference_utility'],
      'quantized_original_total':meta['quantized_optimizer_original_total_utility'],
      'seed_original_total':meta['seed_original_total_utility']}
 rows.append(row);print(json.dumps(row),flush=True)
 methods=[('HiGHS',-1,lambda:solve_milp(p.quantized,10.)),
          ('greedy',-1,lambda:solve_greedy(p.quantized)),
          ('local_search',-1,lambda:solve_local(p.quantized,.2))]
 methods += [('local_SA_NOT_Kaiwu',s,lambda s=s:solve_sa(p.quantized,s,100,4)) for s in (11,22,33)]
 for method,s,fn in methods:
  y,inf=fn();ss=np.r_[2*y-1,1]
  bench.append({'name':name,'method':method,'seed':s,'reward':float(p.quantized.weights@y),
    'H':float(-ss@p.J@ss),'feasible':p.quantized.evaluate(y)['feasible'],
    'original_total_utility':float(full.weights@p.lift(y)),'elapsed_s':inf['elapsed_s'],
    'proven_optimal':inf['optimal'],'is_quantum_result':False})
for dest,records in [(out/'manifest.csv',rows),(results/'classical_comparison.csv',bench)]:
 with dest.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(out/'design.json').write_text(json.dumps({'n_matrices':6,'same_request_trace':True,'first_physical_round':'six matrices, one submission each','optional_stability_rounds':2,'shown_multiple_solutions_are_not_independent_runs':True,'max_binary_variables':96,'min_resolution':1,'quantum_results_pending':True},indent=2))
