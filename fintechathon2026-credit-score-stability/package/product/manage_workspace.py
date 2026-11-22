"""Create a consistent, readable backup of the local case workspace."""
from pathlib import Path
import argparse,json,os,sqlite3
from case_store import CaseStore
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('command',choices=['backup','status'])
p.add_argument('--db',default=os.environ.get('FINTECH_CASE_DB',str(Path(__file__).parent/'runtime/cases.sqlite3')))
p.add_argument('--output',type=Path)
a=p.parse_args();db=Path(a.db)
if not db.exists():p.error('案件工作空间尚未创建，请先启动服务。')
catalog=json.loads((Path(__file__).parent/'case_catalog.json').read_text());store=CaseStore(db,catalog)
if a.command=='backup':
 if not a.output:p.error('backup需要--output参数')
 if a.output.exists():p.error('目标已存在，请使用新的备份文件名。')
 a.output.parent.mkdir(parents=True,exist_ok=True);store.backup(a.output)
 with sqlite3.connect(a.output) as c:
  assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
 print(json.dumps({'backup':str(a.output.resolve()),'bytes':a.output.stat().st_size},ensure_ascii=False))
else:
 s=store.snapshot();print(json.dumps({'cases':len(s['records']),'events':len(s['events']),'statuses':{k:sum(r['status']==k for r in s['records'].values()) for k in ['待复核','核对中','待补充','待复审','已核对']}},ensure_ascii=False,indent=2))
