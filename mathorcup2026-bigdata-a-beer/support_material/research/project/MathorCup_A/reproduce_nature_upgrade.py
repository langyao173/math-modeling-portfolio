#!/usr/bin/env python3
"""Reproduce the source-based extension using the supplied original data.

The package retains complete candidate predictions to make inspection fast.
Changed input data must not silently reuse this study's candidate caches.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'A_work'
UP=WORK/'nature_upgrade'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/A')
    p.add_argument('--stage',choices=['all','audit','baseline','adaptive','gbr','catboost','fusion','sparse','verify'],default='all')
    p.add_argument('--list',action='store_true')
    args=p.parse_args()
    data=args.data.resolve()
    files=['1.csv','2.csv','3.xlsx','4.csv','5.csv','6.csv','7.csv','8.xlsx']
    if not all((data/n).is_file() for n in files):
        p.error('The data directory must contain all eight original attachment files.')
    manifest=json.loads((ROOT/'nature_upgrade_manifest.json').read_text())
    for name,expected in manifest['input_sha256'].items():
        actual=hashlib.sha256((data/name).read_bytes()).hexdigest()
        if actual!=expected:
            p.error(f'{name} differs from the original study input. Do not reuse these caches for changed data.')
    plan={
        'audit':[('data_claim_audit/audit_claims.py',['--data',str(data)]),
                 ('audit_spiking_validation.py',['--workbook',str(WORK/'audit/external/original_supplement.xlsx')])],
        'baseline':[('run_author_split_baseline.py',['--data',str(data)])],
        'adaptive':[('adaptive/run_adaptive.py',['--data',str(data)])],
        'gbr':[('benchmark/run_gbr_challenger.py',['--data',str(data)]),('benchmark/fit_final_gbr.py',['--data',str(data)])],
        'catboost':[('catboost/run_catboost.py',['--data',str(data)])],
        'fusion':[('adaptive/prepare_fusion.py',['--cache-only']),('fusion/run_fusion.py',['--variants','5,7,8','--fit-final'])],
        'sparse':[('sparse_transfer/run_sparse_transfer.py',['--data',str(data)]),('sparse_transfer/verify_and_plot.py',['--data',str(data)])],
        'verify':[('benchmark/audit_fusion.py',[]),('catboost/validate_catboost.py',[])]
    }
    env=os.environ.copy()
    env.update(MATHORCUP_DATA_DIR=str(data),MATHORCUP_Q1_DIR=str(WORK/'q1'),
               MPLCONFIGDIR=str(ROOT/'.mpl_cache'),OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    font=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'
    if font.exists():env['MATHORCUP_CJK_FONT']=str(font)
    log=[]
    stages=list(plan) if args.stage=='all' else [args.stage]
    for stage in stages:
        for script,extra in plan[stage]:
            if args.list:
                print(stage,script,' '.join(extra))
                continue
            print('RUN',stage,script,flush=True)
            started=time.perf_counter()
            subprocess.run([sys.executable,str(UP/script),*extra],cwd=ROOT,env=env,check=True)
            log.append(dict(stage=stage,script=script,seconds=time.perf_counter()-started))
    if not args.list:
        (UP/'reproduction_execution.json').write_text(json.dumps(log,indent=2))
        print('Completed',len(log),'extension steps.')


if __name__=='__main__':main()
