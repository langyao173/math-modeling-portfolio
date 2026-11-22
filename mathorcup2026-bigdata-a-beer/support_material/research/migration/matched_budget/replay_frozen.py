"""Reproduce the fixed new100 validation; never optimize or reselect candidates."""
from pathlib import Path
import sys,argparse,json,shutil
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));import run_scenario as m
import numpy as np
from threadpoolctl import threadpool_limits

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args();out=args.output.resolve();assert out!=HERE,'Use a separate replay output directory';out.mkdir(parents=True,exist_ok=True)
 freeze=json.loads((HERE/'CANDIDATES_FROZEN.json').read_text());assert m.sha(HERE/'frozen_profiles.csv')==freeze['profiles_sha256'];assert m.sha(HERE/'frozen_frontier.csv')==freeze['frontier_sha256']
 for name in ['frozen_profiles.csv','frozen_frontier.csv','CANDIDATES_FROZEN.json','PROTOCOL_LOCK.json','protocol.json']:shutil.copyfile(HERE/name,out/name)
 problem=m.FixedProblem('strict');frame=m.read_csv(out/'frozen_profiles.csv');indices=m.bootstrap(problem.df.tasting_category_fine.to_numpy(str),freeze['scenarios'],freeze['seed']);np.savez_compressed(out/'validation_indices.npz',indices=indices,beer=problem.df.beer.to_numpy())
 with threadpool_limits(limits=1):m.replay(out,problem,frame,indices,'validation',args.workers)
 print('Fixed new100 replay complete; no candidate identity or selection was changed.',flush=True)
if __name__=='__main__':main()
