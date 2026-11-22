"""Versioned local evidence intake. No lending or customer-contact action."""
from pathlib import Path
import hashlib,json,sqlite3

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(x):return hashlib.sha256(canonical(x).encode()).hexdigest()

class IntakeConflict(ValueError):pass

class ContinuityStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as d:
            d.executescript('''PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS episodes (episode TEXT PRIMARY KEY,customer TEXT,account TEXT,epoch TEXT,status TEXT,version INTEGER,last_month INTEGER,last_q REAL,last_content TEXT,owner TEXT);
              CREATE TABLE IF NOT EXISTS observations (snapshot TEXT PRIMARY KEY,fingerprint TEXT,episode TEXT,record TEXT,receipt TEXT);
              CREATE TABLE IF NOT EXISTS workflow (request TEXT PRIMARY KEY,fingerprint TEXT,receipt TEXT);
              CREATE TABLE IF NOT EXISTS actions (seq INTEGER PRIMARY KEY AUTOINCREMENT,episode TEXT,kind TEXT,record TEXT);
              CREATE TRIGGER IF NOT EXISTS immutable_actions_update BEFORE UPDATE ON actions BEGIN SELECT RAISE(ABORT,'append only');END;
              CREATE TRIGGER IF NOT EXISTS immutable_actions_delete BEFORE DELETE ON actions BEGIN SELECT RAISE(ABORT,'append only');END;''')
    def connect(self):
        d=sqlite3.connect(self.path,timeout=10);d.row_factory=sqlite3.Row;return d
    @staticmethod
    def validate(o):
        req=['snapshot_id','customer','account','epoch','month','q','financial','current_dpd','action_type','selected','model_version']
        if any(k not in o for k in req):raise ValueError('证据字段不完整')
        if type(o['month']) is not int:raise ValueError('月份需为整数')
        if not isinstance(o['q'],(float,int)) or not 0<=o['q']<=1:raise ValueError('概率无效')
        if o['action_type'] not in ['unknown','policy','reunderwriting']:raise ValueError('动作类型无效')
        if type(o['selected']) is not bool:raise ValueError('名单标记无效')
        if o['current_dpd'] is not None and o['current_dpd']<0:raise ValueError('逾期天数无效')
        canonical(o) # Reject NaN/Infinity rather than silently entering the service.
    def intake(self,o):
        self.validate(o);fp=sha(o)
        episode=sha([o['customer'],o['account'],o['epoch'],o['model_version']])[:24]
        content=sha({'financial':o['financial'],'dpd':o['current_dpd'],'action':o['action_type']})
        with self.connect() as d:
            d.execute('BEGIN IMMEDIATE')
            previous=d.execute('SELECT * FROM observations WHERE snapshot=?',(o['snapshot_id'],)).fetchone()
            if previous:
                if previous['fingerprint']!=fp:raise IntakeConflict('同一快照编号不能对应不同证据')
                return {**json.loads(previous['receipt']),'idempotent_replay':True}
            old=d.execute('SELECT * FROM episodes WHERE episode=?',(episode,)).fetchone()
            if old and o['month']<=old['last_month']:raise IntakeConflict('迟到快照进入人工校正，不覆盖较新状态')
            missing=any(v is None for v in o['financial'].values()) or len(o['financial'])!=7 or o['current_dpd'] is None
            overdue=o['current_dpd'] is not None and o['current_dpd']>0
            reunderwrite=o['action_type']=='reunderwriting'
            flags=[]
            if overdue:flags.append('已观察逾期，立即转交风险复核')
            if reunderwrite:flags.append('重新资质审查，确认新的参考周期')
            if missing:flags.append('账务缺项，补齐后确认连续性')
            if o['action_type']=='unknown':flags.append('额度动作原因待核实')
            if old and o['q']-old['last_q']>=.02:flags.append('风险上升达到2个百分点')
            if o.get('bounded_streak',0)>=2:flags.append('连续两期触及更新幅度，核对累积变化')
            critical=overdue or reunderwrite or (old and o['q']-old['last_q']>=.02) or o.get('bounded_streak',0)>=2
            actionable=o['selected'] or critical or missing
            if not old and not actionable:
                receipt=dict(episode=episode,operation='observe',status='观察',version=0,flags=flags,evidence_reused=False,idempotent_replay=False)
            else:
                version=1 if not old else old['version']+1
                status='优先复核' if critical else ('待补件' if missing else '待核对')
                if old and old['status']!='已完成':operation='escalate' if critical else 'attach'
                elif old:operation='reopen' if (critical or content!=old['last_content']) else 'observe'
                else:operation='open'
                reused=bool(old and content==old['last_content'] and not critical and not missing)
                if operation=='observe':status='已完成'
                elif old and operation=='attach' and not missing:status=old['status']
                owner=old['owner'] if old else ''
                d.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(episode) DO UPDATE SET status=excluded.status,version=excluded.version,last_month=excluded.last_month,last_q=excluded.last_q,last_content=excluded.last_content',
                    (episode,str(o['customer']),str(o['account']),str(o['epoch']),status,version,o['month'],o['q'],content,owner))
                receipt=dict(episode=episode,operation=operation,status=status,version=version,flags=flags,evidence_reused=reused,idempotent_replay=False)
                d.execute('INSERT INTO actions(episode,kind,record) VALUES (?,?,?)',(episode,operation,canonical(receipt)))
            d.execute('INSERT INTO observations VALUES (?,?,?,?,?)',(o['snapshot_id'],fp,episode,canonical(o),canonical(receipt)))
            return receipt
    def transition(self,request,episode,expected_version,actor,action,note,checks=None,action_confirmed=None,anchor_policy=None):
        payload=dict(request=request,episode=episode,expected_version=expected_version,actor=actor,action=action,note=note,checks=checks,action_confirmed=action_confirmed,anchor_policy=anchor_policy)
        fp=sha(payload)
        if action not in ['claim','submit','approve','return']:raise ValueError('操作无效')
        if not actor or not note:raise ValueError('请填写经办标识和依据')
        with self.connect() as d:
            d.execute('BEGIN IMMEDIATE')
            previous=d.execute('SELECT * FROM workflow WHERE request=?',(request,)).fetchone()
            if previous:
                if previous['fingerprint']!=fp:raise IntakeConflict('请求编号已使用')
                return json.loads(previous['receipt'])
            old=d.execute('SELECT * FROM episodes WHERE episode=?',(episode,)).fetchone()
            if not old or old['version']!=expected_version:raise IntakeConflict('案件版本已更新，请重新加载')
            if action=='claim':
                if old['status'] in ['待复审','已完成']:raise ValueError('当前状态不可领取')
                status='核对中';owner=actor
            elif action=='submit':
                if old['status']!='核对中' or old['owner']!=actor:raise ValueError('需由经办人核对后提交')
                if not isinstance(checks,dict) or any(not str(checks.get(k,'')).strip() for k in ['identity','ledger','action','reference']):raise ValueError('请填写账户、账务、动作及参考周期四项依据')
                if action_confirmed not in ['policy','reunderwriting']:raise ValueError('请核实额度动作类型')
                if anchor_policy!=('retain' if action_confirmed=='policy' else 'refresh'):raise ValueError('动作类型与参考周期处理不一致')
                status='待复审';owner=actor
            else:
                if old['status']!='待复审' or old['owner']==actor:raise ValueError('需由另一人员复审')
                status='已完成' if action=='approve' else '核对中';owner=old['owner']
            r=dict(episode=episode,version=expected_version+1,status=status,owner=owner,actor=actor,note=note,action=action,checks=checks,action_confirmed=action_confirmed,anchor_policy=anchor_policy)
            d.execute('UPDATE episodes SET version=?,status=?,owner=? WHERE episode=?',(r['version'],status,owner,episode))
            d.execute('INSERT INTO workflow VALUES (?,?,?)',(request,fp,canonical(r)))
            d.execute('INSERT INTO actions(episode,kind,record) VALUES (?,?,?)',(episode,action,canonical(r)))
            return r
    def snapshot(self):
        with self.connect() as d:return {'episodes':[dict(r) for r in d.execute('SELECT * FROM episodes')],
            'actions':[dict(r) for r in d.execute('SELECT * FROM actions ORDER BY seq')]}

def evidence_assistant(observation,receipt):
    """Deterministic cited tools; no claim of a connected language model."""
    src=observation['snapshot_id'];q=100*observation['q']
    operation={'observe':'持续观察','open':'建立复核案件','attach':'接续原案并补充证据','escalate':'升级风险核对','reopen':'重新打开案件'}[receipt['operation']]
    facts=[{'text':f'统一评分对应的早期逾期概率为 {q:.2f}%。','source':src+'#q'},
        {'text':'本期进入风险复核名单。' if observation['selected'] else '本期未进入容量名单。','source':src+'#selected'},
        {'text':f'案件处理：{operation}；状态：{receipt["status"]}。','source':receipt['episode']+'#version'}]
    return {'agent':'证据核对助手','mode':'本地结构化工具','facts':facts,'next_checks':receipt['flags'] or ['核对本期账务和额度动作依据'],
        'requires_human_signoff':True,'lending_action':None}
