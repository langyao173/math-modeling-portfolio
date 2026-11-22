"""Local licensed Kaiwu bridge. Default action is dry-run; never auto-spends credits.

API contract checked against Kaiwu documentation 1.4.0 on 2026-09-22.
The licensed SDK was not installed in the delivery environment. Runtime signature
checks are intentional: do not silently adapt unsupported account routing.
"""
from pathlib import Path
import sys,json,argparse,inspect,os,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.qubo import read_matrix,preflight_ising


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matrix',required=True,type=Path)
    p.add_argument('--action',choices=['dry-run','sa','submit','query'],default='dry-run')
    p.add_argument('--task-name',default='QP_CHECK06');p.add_argument('--out',type=Path,default=Path('local_kaiwu_results'))
    p.add_argument('--seed',type=int,default=11);p.add_argument('--project-no',default=None)
    p.add_argument('--routing-confirmed-spqc550',action='store_true')
    p.add_argument('--charge-rule-confirmed',action='store_true')
    p.add_argument('--allow-cloud-submit',action='store_true')
    a=p.parse_args();J=read_matrix(a.matrix);check=preflight_ising(J)
    if not check['valid']:raise SystemExit('Matrix rejected locally: '+','.join(check['issues']))
    if not 5<=len(a.task_name)<=24:raise SystemExit('Task name must be 5-24 characters.')
    print(json.dumps({'action':a.action,'matrix_check':check,'task_name':a.task_name},indent=2))
    if a.action=='dry-run':
        print('No SDK imported, no cloud request sent, no credits consumed.');return
    if sys.version_info[:2]!=(3,10):raise SystemExit('Use Python 3.10 for the licensed SDK/contest execution.')
    try:import kaiwu as kw
    except ImportError as e:raise SystemExit('Install the licensed Kaiwu SDK from its official distribution in Python 3.10.') from e
    a.out.mkdir(parents=True,exist_ok=True)
    checkpoint=a.out/'checkpoint';checkpoint.mkdir(exist_ok=True)
    kw.common.CheckpointManager.save_dir=str(checkpoint.resolve())
    # Credentials are read locally and never saved in reports.
    uid=os.environ.get('KAIWU_USER_ID');code=os.environ.get('KAIWU_SDK_CODE')
    if uid and code:kw.license.init(user_id=uid,sdk_code=code)
    elif hasattr(kw.license,'ensure_license'):kw.license.ensure_license()
    else:raise SystemExit('Initialize your Kaiwu license locally before using this bridge.')
    sdk_version=str(getattr(kw,'__version__','not_exposed'))
    if a.action=='sa':
        kw_args=dict(initial_temperature=100,alpha=.99,cutoff_temperature=.001,iterations_per_t=10,
                     size_limit=20,flag_evolution_history=True,rand_seed=a.seed,process_num=1)
        sig=inspect.signature(kw.classical.SimulatedAnnealingOptimizer)
        if not all(k in sig.parameters for k in kw_args):raise SystemExit('Unsupported SA signature. Inspect the installed SDK documentation.')
        worker=kw.classical.SimulatedAnnealingOptimizer(**kw_args)
        started=time.perf_counter();solutions=worker.solve(J,negtail_flip=False,sort_solutions=False)
        elapsed=time.perf_counter()-started
        np.savetxt(a.out/'kaiwu_sa_spins.csv',solutions,delimiter=',',fmt='%d')
        info={'backend':'Kaiwu_SA_classical','sdk_version':sdk_version,'elapsed_s':elapsed,'settings':kw_args,
              'n_returned_rows':len(solutions),'physical_run':False}
        (a.out/'kaiwu_sa_run.json').write_text(json.dumps(info,indent=2))
        if hasattr(worker,'get_ha_history'):
            (a.out/'kaiwu_sa_history.json').write_text(json.dumps(worker.get_ha_history(),default=lambda x:np.asarray(x).tolist()))
        print(json.dumps(info,indent=2));return
    if a.action=='submit' and not (a.allow_cloud_submit and a.routing_confirmed_spqc550 and a.charge_rule_confirmed):
        raise SystemExit('Submission blocked. First confirm account/device routing and charging; pass all three explicit consent flags.')
    sig=inspect.signature(kw.cim.CIMOptimizer)
    kwargs=dict(task_name=a.task_name,wait=False,interval=1,task_mode='optimization')
    if a.project_no is not None:kwargs['project_no']=a.project_no
    if not all(k in sig.parameters for k in kwargs):raise SystemExit('Unsupported CIMOptimizer signature; no task submitted.')
    # The documented constructor has no machine_name parameter. Do not invent one.
    identity=a.out/(a.task_name+'_submission.json')
    if a.action=='submit' and identity.exists():raise SystemExit('This task has a submission record. Use --action query, not another submission.')
    worker=kw.cim.CIMOptimizer(**kwargs)
    if a.action=='query':
        result=worker.get_task_result(J)
        (a.out/(a.task_name+'_provider_result.json')).write_text(json.dumps(result,default=lambda x:np.asarray(x).tolist(),indent=2))
        print('Provider result saved without intentionally creating a new task.');return
    info={'task_name':a.task_name,'matrix_name':a.matrix.name,'sdk_version':sdk_version,'action':'attempting_submission',
          'routing_confirmed_by_user':'SPQC-550','charge_rule_confirmed_by_user':True,
          'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    identity.write_text(json.dumps(info,indent=2)) # record before request; failures must not auto-retry/charge again
    started=time.perf_counter()
    result=worker.solve(J,negtail_flip=False,sort_solutions=False)
    info.update(action='submission_call_returned',submission_call_elapsed_s=time.perf_counter()-started)
    identity.write_text(json.dumps(info,indent=2))
    if result is not None:
        np.savetxt(a.out/(a.task_name+'_spins.csv'),np.atleast_2d(result),delimiter=',',fmt='%d')
    print('Submission call returned. Keep the task name and matrix unchanged; obtain the provider report and query rather than resubmit.')
if __name__=='__main__':main()
