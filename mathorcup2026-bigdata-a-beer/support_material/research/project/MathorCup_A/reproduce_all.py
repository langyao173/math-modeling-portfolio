#!/usr/bin/env python3
"""Reproduce the executed MathorCup A study in the supplied directory structure.

The full run retrains all models and overwrites derived result files in A_work.
Original attachments are read only. No external service is needed for analysis.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'A_work').exists() else HERE.parent
WORK=ROOT/'A_work'

def jobs(data):
    return {
      'audit': [('audit/audit_data.py',['--source',str(data),'--output',str(WORK/'audit')])],
      'q1': [('q1/run_q1.py',['--data',str(data),'--output',str(WORK/'q1')]),('q1/fit_knn_variant.py',['--data',str(data)])],
      'q2': [('q2/run_q2.py',['--data',str(data),'--output',str(WORK/'q2'),'--bootstrap','400','--stratified','200']),('q2/make_q2_report.py',['--data',str(data),'--output',str(WORK/'q2')])],
      'q3': [('q3/run_q3.py',['--data',str(data),'--out',str(WORK/'q3')])]+[(f'q3/{n}.py',[]) for n in ['evaluate_extensions','transfer_profiles','transfer_controls','audit_and_figures','sensory_profiles','repair_stale','audit_certificates']],
      'validation': [(f'audit/model_validation/{n}.py',[]) for n in ['validate_models','strict_missingness','strict_knn_missingness']]+[('audit/verify_final_transfer.py',[])],
      'figures': [('make_analysis_figures.py',[])],
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'data'/'A')
    ap.add_argument('--stage',choices=['all','audit','q1','q2','q3','validation','figures'],default='all')
    ap.add_argument('--list',action='store_true',help='Show execution plan without running it.')
    args=ap.parse_args();data=args.data.resolve()
    expected=['1.csv','2.csv','3.xlsx','4.csv','5.csv','6.csv','7.csv','8.xlsx']
    if not all((data/f).is_file() for f in expected):
        ap.error('Data directory must contain all eight original attachment files.')
    plan=jobs(data);names=list(plan) if args.stage=='all' else [args.stage]
    env=os.environ.copy()
    env.update(MATHORCUP_DATA_DIR=str(data),MATHORCUP_Q1_DIR=str(WORK/'q1'),
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str(ROOT/'.mpl_cache'))
    font=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'
    if font.exists():env['MATHORCUP_CJK_FONT']=str(font)
    log=[];start=time.time()
    for stage in names:
        for file,extra in plan[stage]:
            command=[sys.executable,str(WORK/file),*extra]
            if args.list:
                print(stage,':',file,' '.join(extra));continue
            print('\nRUN',stage,file,flush=True);t=time.time()
            subprocess.run(command,env=env,cwd=ROOT,check=True)
            log.append(dict(stage=stage,script=file,seconds=round(time.time()-t,3)))
    if not args.list:
        (WORK/'reproduction_execution.json').write_text(json.dumps(dict(total_seconds=time.time()-start,runs=log),indent=2))
        print('Completed',len(log),'analysis steps.')

if __name__=='__main__':main()
