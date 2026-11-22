from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[1]
O=R/'results/stage3/verified'
rows=[];records=[]
for p in sorted(O.glob('QP_APP_A_??.json')):
 r=json.loads(p.read_text());name=r['name'];m=json.loads((R/'upload/03_application_batch'/f'{name}.json').read_text())
 snapshot=json.loads((R/'results/stage2/application_batch'/f'{name}_snapshot.json').read_text())
 raw=next(x for x in r['rows'] if x['result_index']==r['selected_result_index'])
 plans={'raw_selected':raw['lifted_full_bits'],'after_classical_polish':r['polished_lifted_bits'],'classical_seed':m['lifting']['seed']}
 for kind,bits in plans.items():
  selected=[]
  for i,b in enumerate(bits):
   if not b:continue
   label=snapshot['labels'][i];route=snapshot['candidate_routes'][i]
   for rid in label['requests']:
    picks=[j for j,s in enumerate(route) if s['rid']==rid and s['pickup']]
    drops=[j for j,s in enumerate(route) if s['rid']==rid and not s['pickup']]
    assert len(picks)==len(drops)==1 and picks[0]<drops[0]
   selected.append({'full_candidate_index':i,**label,'route':route})
   rows.append({'name':name,'selection_stage':kind,'full_candidate_index_zero_based':i,
                'vehicle_id':label['vehicle'],'new_request_ids':json.dumps(label['requests']),
                'candidate_original_utility':snapshot['weights'][i],
                'ordered_stops':json.dumps(route)})
  records.append({'name':name,'stage':kind,'source_epoch_s':m['source_epoch'],
                 'checked':'unique new requests and vehicles; exactly one pickup before dropoff for each new request',
                 'not_checked':'snapshot does not contain initial vehicle state/ETA; no replay of live route timing',
                 'plans':selected})
with (O/'reconstructed_assignments.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'reconstructed_assignments.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
print(len(rows),'selected candidate plans checked')
