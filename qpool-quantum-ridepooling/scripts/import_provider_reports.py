"""Import the provider PDFs without submitting a task or using hardware credits."""
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qpool.provider_reports import verify_provider_report


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--reports',type=Path,default=ROOT/'results/stage3/provider_reports')
    ap.add_argument('--models',type=Path,default=ROOT/'upload/03_application_batch')
    ap.add_argument('--output',type=Path,default=ROOT/'results/stage3/verified')
    args=ap.parse_args()
    paths=sorted(args.reports.glob('*.pdf'))
    if not paths:
        raise SystemExit('No PDF reports found.')
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[];details=[];seen_ids=set();complete=[]
    for p in paths:
        report=verify_provider_report(p,args.models)
        task_id=report['provider']['task_id']
        if task_id in seen_ids:
            raise ValueError('The same task ID appears twice; do not count it as a repeat.')
        seen_ids.add(task_id)
        summary=report['summary'];summaries.append(summary);complete.append(report)
        stem=report['provider']['task_name']
        if not all(c.isalnum() or c in '_-' for c in stem):
            raise ValueError('Unsupported task name for local output.')
        (args.output/(stem+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        with (args.output/(stem+'_spins.csv')).open('w',newline='') as f:
            csv.writer(f).writerows(r['spins'] for r in report['provider']['rows'])
        for r in report['rows']:
            details.append({
                'name':report['name'],'task_name':stem,'task_id':task_id,
                'result_index':r['result_index'],'reported_H':r['reported_H'],
                'recomputed_H':r['ising_energy'],'energy_residual':r['energy_residual'],
                'raw_feasible':r['raw_feasible'],'raw_conflicts':r['raw_conflicts'],
                'hits_known_quantized_optimum':r['hits_known_quantized_optimum'],
                'raw_quantized_reward':r['raw_reward'],
                'raw_lifted_original_utility':r['lifted_raw_original_utility'],
                'raw_new_request_count':r.get('new_request_count',''),
                'polished_lifted_original_utility':r.get('tie_polished_full_original_utility',''),
                'polish_elapsed_s':r.get('tie_polish',{}).get('elapsed_s',''),
                'classically_repaired_reward':r['classically_repaired_reward'],
                'spins':json.dumps(r['spins']), 'bits':json.dumps(r['bits']),
                'lifted_bits':json.dumps(r['lifted_full_bits']),
            })
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    write_csv(args.output/'task_summary.csv',summaries)
    write_csv(args.output/'all_60_returned_candidates.csv',details)
    agg={
        'provider_reports':len(summaries), 'distinct_provider_task_ids':len(seen_ids),
        'returned_candidates':sum(s['returned_rows'] for s in summaries),
        'matching_vector_energy_pairs':sum(s['energy_checks_passed'] for s in summaries),
        'tasks_with_known_quantized_optimum':sum(abs(s['absolute_H_gap'])<1e-8 for s in summaries),
        'selected_raw_assignments_feasible':sum(s['raw_feasible_rows']>0 for s in summaries),
        'raw_feasible_candidates':sum(s['raw_feasible_rows'] for s in summaries),
        'optimal_returned_candidates':sum(s['optimal_rows'] for s in summaries),
        'conditional_original_optimum_after_classical_polish':sum(abs(s['conditional_original_gap'])<1e-8 for s in summaries),
        'complete_original_optimum_after_classical_polish':sum(abs(s['full_original_gap'])<1e-8 for s in summaries),
        'report_server_span_within_epoch':sum(s['server_span_within_epoch'] for s in summaries),
        'independent_runs_per_instance':1,
        'candidate_counts_are_not_success_probabilities':True,
        'provider_sampling_count':'N/A in all six reports',
        'full_day_hardware_replay_performed':False,
        'new_hardware_submissions_by_this_script':0,
    }
    (args.output/'aggregate.json').write_text(json.dumps(agg,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AGGREGATE',json.dumps(agg,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
