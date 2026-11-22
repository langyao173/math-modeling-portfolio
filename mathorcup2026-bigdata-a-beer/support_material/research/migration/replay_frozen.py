"""Repeat the frozen200-scenario validation without reselecting candidates."""
from pathlib import Path
import argparse,sys,shutil,json
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import run_scenario as m
from threadpoolctl import threadpool_limits

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 source=P/'all11';freeze=json.loads((source/'CANDIDATES_FROZEN.json').read_text());assert m.sha(source/'frozen_validation_profiles.csv')==freeze['profile_sha256']
 for name in ['frozen_validation_profiles.csv','development_frontier.csv','CANDIDATES_FROZEN.json','PRIMARY_ANALYSIS_LOCK.json']:
  shutil.copyfile(source/name,out/name)
 problem=m.FixedProblem('strict');frame=m.read_csv(out/'frozen_validation_profiles.csv');indices=m.bootstrap(problem.df.tasting_category_fine.to_numpy(str),200,m.FRESH_SEED)
 with threadpool_limits(limits=1):m.replay(out,problem,frame,indices,'validation',args.workers)
 print('Complete: output contains raw draws and stability of the original frozen profiles.',flush=True)
if __name__=='__main__':main()
