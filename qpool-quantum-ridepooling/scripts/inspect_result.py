from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qpool.results import read_solutions,inspect_solutions

def main():
    p=argparse.ArgumentParser();p.add_argument('--meta',required=True,type=Path)
    p.add_argument('--solutions',required=True,type=Path);p.add_argument('--representation',required=True,choices=['ising','qubo'])
    p.add_argument('--source',default='unverified_external');p.add_argument('--out',type=Path,default=Path('result_inspection.json'))
    a=p.parse_args();r=inspect_solutions(a.meta,read_solutions(a.solutions),a.representation,a.source)
    a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
