"""Retrieve the unmodified public example into a private local research folder.

Do not redistribute the downloaded NYU repository/data or modify its software
under the displayed non-commercial research license without the required rights.
This independent project does not bundle the original NYU source or data files.
"""
from pathlib import Path
from urllib.request import urlopen,Request
import argparse,csv,io,hashlib,json,time

COMMIT='fe3af6015fe651d52f911b6ab96f30cbb0b9de4f'
ROOT=f'https://raw.githubusercontent.com/BUILTNYU/ridepooling/{COMMIT}/'
EXPECTED={'SiouxFalls_TTmatrix.csv':'9dc22ee6d2ff8ecd2106747e4593976e014afae6',
          'SiouxFalls_stops.csv':'70e6ad5aba2ce7dfb6cad7d73a80da315a250f84',
          'SiouxFalls_requests.csv':'1402b6661678b706712169ff7d4d9be89997664e',
          'LICENSE':'88a9bc2e761846b0f278a0341edde53f74c48890'}
def git_blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('data/private_example'))
    p.add_argument('--accept-source-license',action='store_true');p.add_argument('--prefix',type=int,default=240)
    a=p.parse_args()
    if not a.accept_source_license:raise SystemExit('Read the upstream LICENSE and acknowledge internal-research use with --accept-source-license.')
    if a.prefix<1:raise SystemExit('Prefix must be positive.')
    a.out.mkdir(parents=True,exist_ok=True);files={}
    for name in EXPECTED:
        try:
            with urlopen(Request(ROOT+name,headers={'User-Agent':'QPool research reproducibility'}),timeout=45) as r:data=r.read()
        except Exception as e:raise SystemExit(f'Failed to retrieve {name}: {e}. No synthetic fallback was generated.') from e
        if git_blob(data)!=EXPECTED[name]:raise SystemExit('Upstream byte identity mismatch for '+name)
        files[name]=data;(a.out/name).write_bytes(data)
    stops=list(csv.DictReader(io.StringIO(files['SiouxFalls_stops.csv'].decode())))
    lookup={(round(float(r['x']),8),round(float(r['y']),8)):int(r['id']) for r in stops}
    source=list(csv.DictReader(io.StringIO(files['SiouxFalls_requests.csv'].decode())))[:a.prefix]
    rows=[]
    for i,r in enumerate(source):
        o=(round(float(r['triprequest_origin_location_lon']),8),round(float(r['triprequest_origin_location_lat']),8))
        d=(round(float(r['triprequest_destination_location_lon']),8),round(float(r['triprequest_destination_location_lat']),8))
        if o not in lookup or d not in lookup:raise SystemExit('Coordinates do not exactly match source stops; no undocumented snapping allowed.')
        rows.append({'request_id':i,'origin':lookup[o],'destination':lookup[d],'arrival':int(r['triprequest_requesttime'])})
    with (a.out/'requests.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['request_id','origin','destination','arrival']);w.writeheader();w.writerows(rows)
    (a.out/'provenance.json').write_text(json.dumps({'repository':'BUILTNYU/ridepooling','commit':COMMIT,'requests_prefix':len(rows),
        'source_interpretation':'public example; not claimed to be observed real-service demand','transformation':'exact coordinate-to-stop mapping, unchanged arrival order'},indent=2))
    print('Retrieved pinned source and prepared',len(rows),'requests. Keep this directory private unless redistribution rights are clarified.')
if __name__=='__main__':main()
