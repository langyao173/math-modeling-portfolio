"""Local case service: transactional workflow, optimistic locking, idempotent writes.

Scores and observations are immutable files. Only operating records enter SQLite.
Operator aliases identify workflow actions; institutional authentication is an adapter boundary.
"""
from pathlib import Path
from datetime import datetime, timezone, date
import hashlib, json, sqlite3, uuid

STATUSES=['待复核','核对中','待补充','待复审','已核对']
ACTORS=['复核01','运营01','客服01','复审01']
CHECKS={'identity':'账户与观察期','ledger':'账务金额与入账','action':'额度动作依据','reference':'参考周期处理'}
ACTIONS={'save':'保存办理','claim':'领取案件','assign':'分派案件','submit':'提交复审','approve':'完成复审','return':'退回补充','reopen':'重新办理','import':'导入记录'}
class WorkflowError(ValueError):pass
class Conflict(WorkflowError):
    def __init__(self,current):self.current=current;super().__init__('案件已有较新版本，请加载后继续。')

def blank(row):
    return dict(case_key=row['case_key'],customer_key=row['customer_key'],account_key=row['account_key'],source_row_id=row['row_id'],reference_epoch=row['reference_epoch'],version=0,owner='风险复核岗',assignee='',status='待复核',result='待核对',topic='额度动作核验',due='',receipt='',note='',action_type='unknown',action_initiator='unknown',qualification_change='unknown',action_evidence_id='',action_evidence_month='',score_month=int(row.get('anchor_month',0)),anchor_policy='pending',checks={k:{'status':'pending','reference':''} for k in CHECKS},submitted_by='',reviewed_by='',review_note='',created_at='',updated_at='')

def clean(row,source=None):
    source=source or {};r=blank(row)
    for k in ['owner','assignee','status','result','topic','due','receipt','note','action_type','action_initiator','qualification_change','action_evidence_id','action_evidence_month','anchor_policy','submitted_by','reviewed_by','review_note','created_at','updated_at']:
        if source.get(k) is not None:r[k]=str(source[k]).strip()[:12000 if k=='note' else 1000]
    v=source.get('version',0);r['version']=v if type(v) is int and v>=0 else 0
    if r['status'] not in STATUSES:raise WorkflowError('办理状态无效')
    if r['assignee'] and r['assignee'] not in ACTORS:raise WorkflowError('请选择工作台中的经办人')
    if r['due']:
        try:date.fromisoformat(r['due'])
        except ValueError:raise WorkflowError('跟进日期格式无效')
    if r['action_type'] not in ['unknown','policy','reunderwriting','other']:raise WorkflowError('动作分类无效')
    if r['anchor_policy'] not in ['pending','retain','refresh','handoff']:raise WorkflowError('参考周期处理无效')
    checks=source.get('checks') or {}
    if not isinstance(checks,dict):raise WorkflowError('证据清单无效')
    for k in CHECKS:
        c=checks.get(k)
        if isinstance(c,dict):r['checks'][k]={'status':c.get('status') if c.get('status') in ['pending','needs','verified'] else 'pending','reference':str(c.get('reference') or '').strip()[:600]}
    return r

def blockers(r):
    b=[]
    if not r['assignee']:b.append('指定经办人')
    if not r['note']:b.append('填写办理说明')
    if not r['receipt']:b.append('填写凭证或交接回执')
    if r['result']=='待核对':b.append('选择核对结论')
    handoff=r['result']=='交由既有流程处理'
    for k in (['identity','ledger'] if handoff else CHECKS):
        if r['checks'][k]['status']!='verified' or not r['checks'][k]['reference']:b.append('核对'+CHECKS[k]+'并填写依据')
    if not handoff and r['action_type']=='unknown':b.append('确认额度动作分类')
    if r['anchor_policy']=='pending':b.append('选择参考周期处理')
    if r['action_type']=='reunderwriting' and r['anchor_policy']!='refresh':b.append('重新资质审查应转入参考周期复核')
    if r['action_type']=='unknown' and r['anchor_policy']!='handoff':b.append('动作原因待确认时转交补充依据')
    if r['action_initiator'] not in ['unknown','bank','customer','technical']:b.append('确认动作发起方')
    if r['qualification_change'] not in ['unknown','unchanged','updated']:b.append('确认资质信息状态')
    if r['anchor_policy']=='retain':
        if r['action_type']!='policy' or r['action_initiator']!='bank' or r['qualification_change']!='unchanged':
            b.append('沿用参考周期须核实银行经营动作及资质信息未更新')
        if not r['action_evidence_id']:b.append('登记原始动作凭证编号')
        try:
            month=int(r['action_evidence_month'])
            if str(month)!=r['action_evidence_month'] or month>r['score_month']:raise ValueError()
        except (ValueError,TypeError):b.append('确认动作依据在评分观察月已可用')
    if r['qualification_change']=='updated' and r['anchor_policy']!='refresh':b.append('资质信息更新须转入新参考周期复核')
    return b

def transition(row,before,incoming,action,actor,now):
    if actor not in ACTORS:raise WorkflowError('经办标识无效')
    old=clean(row,before);r=clean(row,{**old,**incoming})
    if action not in ACTIONS:raise WorkflowError('办理操作无效')
    if action in ['approve','return']:
        if old['status']!='待复审':raise WorkflowError('请先提交复审')
        if actor==old['submitted_by']:raise WorkflowError('请由另一位复核人员完成复审')
        if not r['review_note']:raise WorkflowError('请填写复审意见')
        memo=r['review_note'];r=clean(row,old);r['review_note']=memo
        if action=='approve':
            b=blockers(r)
            if b:raise WorkflowError('；'.join(b))
            r['status']='已核对';r['reviewed_by']=actor
        else:r['status']='核对中';r['reviewed_by']='';r['submitted_by']=''
    elif action=='reopen':
        if old['status']!='已核对':raise WorkflowError('仅已完成案件可重新办理')
        if not r['review_note']:raise WorkflowError('请填写重新办理的原因')
        r['status']='核对中';r['submitted_by']='';r['reviewed_by']=''
    else:
        if old['status'] in ['待复审','已核对']:raise WorkflowError('请先退回或重新办理后再修改')
        if action=='claim':r['assignee']=actor;r['status']='核对中'
        if action=='assign' and not r['assignee']:raise WorkflowError('请选择接续经办人')
        if action=='submit':
            b=blockers(r)
            if b:raise WorkflowError('；'.join(b))
            r['status']='待复审';r['submitted_by']=actor;r['reviewed_by']='';r['review_note']=''
        elif r['status'] not in ['待复核','核对中','待补充']:raise WorkflowError('请通过“提交复审”完成办理')
        if r['status']=='待补充' and (not r['due'] or not r['note']):raise WorkflowError('待补充案件需要跟进日期和补充事项')
    r['version']=old['version']+1;r['created_at']=old['created_at'] or now;r['updated_at']=now
    changed=[k for k in r if k not in ['version','created_at','updated_at'] and r[k]!=old[k]]
    ev=dict(case_key=row['case_key'],customer_key=row['customer_key'],source_row_id=row['row_id'],action=action,action_label=ACTIONS[action],actor=actor,from_status=old['status'],status=r['status'],owner=r['owner'],assignee=r['assignee'],version=r['version'],updated_at=now,changed=changed,note=r['review_note'] if action in ['approve','return','reopen'] else r['note'],topic=r['topic'])
    return r,ev

class CaseStore:
    def __init__(self,path,catalog):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.catalog=catalog
        with self.connect() as db:
            db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS cases (case_key TEXT PRIMARY KEY, version INTEGER NOT NULL, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, case_key TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL, UNIQUE(case_key,version));
            CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'events are append-only'); END;''')
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10);db.row_factory=sqlite3.Row;return db
    def _snapshot(self,db):
        return {'records':{r['case_key']:json.loads(r['record']) for r in db.execute('SELECT * FROM cases')},'events':[json.loads(r['payload']) for r in db.execute('SELECT payload FROM events ORDER BY rowid')]}
    def snapshot(self):
        with self.connect() as db:return self._snapshot(db)
    def commit(self,request):
        rid=request.get('request_id');items=request.get('items');actor=request.get('actor')
        if not isinstance(rid,str) or not 8<=len(rid)<=100:raise WorkflowError('请求编号无效')
        if not isinstance(items,list) or not 1<=len(items)<=30:raise WorkflowError('每次提交1至30个案件')
        if any(not isinstance(i,dict) for i in items):raise WorkflowError('案件请求无效')
        keys=[i.get('case_key') for i in items]
        if len(set(keys))!=len(keys):raise WorkflowError('一次请求不能重复包含同一案件')
        fingerprint=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prev=db.execute('SELECT * FROM requests WHERE request_id=?',(rid,)).fetchone()
            if prev:
                if prev['fingerprint']!=fingerprint:raise WorkflowError('请求编号已用于不同内容')
                return {**self._snapshot(db),'replayed':True}
            changes=[];now=datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')
            for item in items:
                row=self.catalog.get(item['case_key'])
                if not row:raise WorkflowError('案件不存在')
                existing=db.execute('SELECT record FROM cases WHERE case_key=?',(row['case_key'],)).fetchone()
                old=json.loads(existing['record']) if existing else blank(row)
                if type(item.get('expected_version')) is not int or item['expected_version']!=old['version']:raise Conflict(old)
                incoming=item.get('record')
                if not isinstance(incoming,dict):raise WorkflowError('办理材料无效')
                record,event=transition(row,old,incoming,item.get('action'),actor,now)
                event.update(event_id=str(uuid.uuid4()),request_id=rid)
                changes.append((record,event))
            for r,e in changes:
                db.execute('INSERT INTO cases VALUES (?,?,?) ON CONFLICT(case_key) DO UPDATE SET version=excluded.version,record=excluded.record',(r['case_key'],r['version'],json.dumps(r,ensure_ascii=False)))
                db.execute('INSERT INTO events VALUES (?,?,?,?)',(e['event_id'],e['case_key'],e['version'],json.dumps(e,ensure_ascii=False)))
            db.execute('INSERT INTO requests VALUES (?,?,?)',(rid,fingerprint,json.dumps([r['case_key'] for r,e in changes])))
            result=self._snapshot(db);db.commit();return {**result,'replayed':False}
    def backup(self,target):
        with self.connect() as source,sqlite3.connect(target) as dest:source.backup(dest)
