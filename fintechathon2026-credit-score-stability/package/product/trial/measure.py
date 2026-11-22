"""Blind review and practitioner-result analysis. Automation excluded by default."""
from pathlib import Path
import argparse,csv,hashlib,json,sqlite3,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def blind(n):return hashlib.sha256(('043-review-'+str(n)).encode()).hexdigest()[:16]

def export(c,out):
    out.mkdir(parents=True,exist_ok=True)
    rows=pd.read_sql_query("SELECT a.id,a.answers,a.task_id,t.case_id FROM attempts a JOIN tasks t ON t.id=a.task_id JOIN participants p ON p.id=t.participant WHERE p.kind='participant' ORDER BY a.id",c)
    cases={x['id']:x for x in json.loads((ROOT/'cases.json').read_text())};packet=[]
    for r in rows.to_dict('records'):
        packet.append(dict(blind_id=blind(r['id']),case_facts=cases[r['case_id']],submitted_material=json.loads(r['answers'])))
    (out/'blind_review_packets.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2))
    with (out/'blind_grades.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['blind_id','completeness','appropriateness','decision','note','reviewer']);w.writeheader()
        for x in packet:w.writerow(dict(blind_id=x['blind_id']))
    print(json.dumps(dict(real_submissions=len(packet),output=str(out)),ensure_ascii=False))

def apply_grades(c,path):
    ids={blind(r[0]):r[0] for r in c.execute('SELECT id FROM attempts')};n=0
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            if r['blind_id'] not in ids:raise ValueError('Unknown blind id')
            com,app=int(r['completeness']),int(r['appropriateness']);decision=r['decision']
            if com not in range(3) or app not in range(3) or decision not in ['approved','return'] or not r['reviewer'].strip():raise ValueError('Invalid review')
            aid=ids[r['blind_id']]
            c.execute('INSERT OR REPLACE INTO reviews VALUES(?,?,?,?,?,?,?)',(aid,com,app,decision,r['note'],r['reviewer'],time.time()))
            if decision=='return':
                # Only the current attempt can reopen a case; an old review must
                # not overwrite a later submitted version.
                c.execute("UPDATE tasks SET status='returned' WHERE id=(SELECT task_id FROM attempts WHERE id=?) AND version=(SELECT version FROM attempts WHERE id=?)",(aid,aid))
            n+=1
    c.commit();print(json.dumps(dict(applied=n)))

def analyze(c,out,B=2000):
    out.mkdir(parents=True,exist_ok=True)
    counts=dict(c.execute('SELECT kind,count(*) FROM participants GROUP BY kind').fetchall())
    d=pd.read_sql_query('''SELECT p.id participant,p.role,p.experience,t.id task,t.case_id,t.arm,t.status,t.started,t.active,t.version,a.objective_pass,a.complete_fields,r.completeness,r.appropriateness,r.decision
    FROM tasks t JOIN participants p ON p.id=t.participant
    LEFT JOIN attempts a ON a.task_id=t.id AND a.version=t.version
    LEFT JOIN reviews r ON r.attempt_id=a.id
    WHERE p.kind='participant' ORDER BY p.id,t.position''',c)
    receipt=dict(human_participants=counts.get('participant',0),automation_participants_excluded=counts.get('automation',0),analysis_population='all assigned tasks of real participants; unsubmitted tasks have zero completed output',generalization='fixed 18 cases; participant-cluster bootstrap',primary='C vs B; A is secondary',measured_benefit_available=False)
    if not len(d):
        receipt['status']='等待从业者试验；未生成工时、正确率或商业收益数值。'
        (out/'measurement_status.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False));return
    d['correct']=d.objective_pass.fillna(0).eq(1).astype(int)
    d['qualified']=(d.correct.eq(1)&d.complete_fields.fillna(0).eq(1)&d.completeness.eq(2)&d.appropriateness.eq(2)&d.decision.eq('approved')).astype(int)
    d['submitted']=d.version.gt(0).astype(int);d['graded']=d.completeness.notna().astype(int);d['reworked']=d.version.gt(1).astype(int)
    grouped=d.groupby('arm').agg(assigned=('task','size'),started=('started','count'),submitted=('submitted','sum'),graded=('graded','sum'),correct=('correct','sum'),qualified=('qualified','sum'),active_seconds=('active','sum'),reworked=('reworked','sum')).reindex(list('ABC'))
    grouped['objective_correct_rate']=grouped.correct/grouped.assigned;grouped['qualified_rate']=grouped.qualified/grouped.assigned
    grouped['active_seconds_per_qualified']=grouped.active_seconds/grouped.qualified.replace(0,np.nan)
    grouped.to_csv(out/'condition_results.csv');d.to_csv(out/'task_measurements.csv',index=False)
    ppl=sorted(d.participant.unique());rng=np.random.default_rng(430410);w=rng.multinomial(len(ppl),np.full(len(ppl),1/len(ppl)),size=B)
    stats={}
    for arm in 'ABC':
        sub=d[d.arm.eq(arm)].groupby('participant').agg(n=('task','size'),correct=('correct','sum'),qualified=('qualified','sum'),active=('active','sum')).reindex(ppl,fill_value=0)
        v=w@sub.to_numpy();stats[arm]=dict(correct=v[:,1]/v[:,0],qualified=v[:,2]/v[:,0],cost=np.divide(v[:,3],v[:,2],out=np.full(B,np.nan),where=v[:,2]>0))
    ci=[]
    for rival in ['B','A']:
        for metric in ['correct','qualified','cost']:
            vals=stats['C'][metric]-stats[rival][metric];finite=np.isfinite(vals)
            ci.append(dict(comparison='C-'+rival,metric=metric,ci95_low=float(np.quantile(vals[finite],.025)) if finite.any() else None,ci95_high=float(np.quantile(vals[finite],.975)) if finite.any() else None,valid_bootstrap_draws=int(finite.sum()),repetitions=B))
    (out/'paired_intervals.json').write_text(json.dumps(ci,ensure_ascii=False,indent=2,allow_nan=False))
    balance=d.groupby(['role','experience','arm']).size().reset_index(name='tasks');balance.to_csv(out/'allocation_balance.csv',index=False)
    receipt.update(measured_benefit_available=bool(d.graded.sum()),submitted_tasks=int(d.submitted.sum()),graded_tasks=int(d.graded.sum()),qualified_tasks=int(d.qualified.sum()),incomplete_participants=int((d.groupby('participant').submitted.sum()<9).sum()))
    (out/'measurement_status.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['export','grade','analyze']);a.add_argument('--db',type=Path,default=ROOT/'state/trial.sqlite');a.add_argument('--out',type=Path,default=ROOT/'measurements');a.add_argument('--grades',type=Path);args=a.parse_args()
    if not args.db.exists():raise SystemExit('Run server.py once to initialize the local trial database.')
    with sqlite3.connect(args.db) as c:
        if args.action=='export':export(c,args.out)
        elif args.action=='grade':apply_grades(c,args.grades)
        else:analyze(c,args.out)
