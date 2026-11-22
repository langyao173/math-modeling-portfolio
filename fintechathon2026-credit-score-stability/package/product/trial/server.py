"""Local, anonymous three-interface trial. Standard-library server; no bank access.

Run: python server.py --port 8768
Functional checks use --db state/automation.sqlite and kind=automation.
The server never serves answer keys, source customer IDs, or future outcomes.
"""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import argparse,hashlib,itertools,json,mimetypes,secrets,sqlite3,time

ROOT=Path(__file__).resolve().parent
CASES={x['id']:x for x in json.loads((ROOT/'cases.json').read_text())}
KEYS=json.loads((ROOT/'answer_key.json').read_text())
PERMS=list(itertools.permutations('ABC'))
DB=ROOT/'state/trial.sqlite'

def connection():
    c=sqlite3.connect(DB,timeout=10);c.row_factory=sqlite3.Row;c.execute('PRAGMA foreign_keys=ON');return c

def init():
    DB.parent.mkdir(parents=True,exist_ok=True)
    with connection() as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS participants(id INTEGER PRIMARY KEY, token TEXT UNIQUE, role TEXT, experience TEXT, kind TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, participant INTEGER REFERENCES participants(id), case_id TEXT, arm TEXT, position INTEGER, status TEXT, started REAL, last_tick REAL, active REAL DEFAULT 0, version INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, task_id TEXT REFERENCES tasks(id), version INTEGER, request_id TEXT UNIQUE, answers TEXT, objective_pass INTEGER, complete_fields INTEGER, submitted REAL, active_total REAL, wall_total REAL, UNIQUE(task_id,version));
        CREATE TABLE IF NOT EXISTS reviews(attempt_id INTEGER PRIMARY KEY REFERENCES attempts(id), completeness INTEGER, appropriateness INTEGER, decision TEXT, note TEXT, reviewer TEXT, reviewed REAL);''')

def assignment(seq):
    panel=(seq//6)%2;perm=PERMS[seq%6];rows=[]
    for block in range(3):
        # Same nine cases recur across six permutations, twice in every arm.
        # Every participant sees each case once and all three case strata per arm.
        order=list(range(3));rotation=(seq+block)%3;order=order[rotation:]+order[:rotation]
        for stratum in order:
            case=stratum*6+panel*3+block+1
            rows.append((f'T{case:02d}',perm[block]))
    return rows

def auth(c,token):
    p=c.execute('SELECT * FROM participants WHERE token=?',(token,)).fetchone()
    if not p:raise ValueError('会话不存在，请重新登记。')
    return p

def session(c,token):
    p=auth(c,token);tasks=[dict(x) for x in c.execute('SELECT * FROM tasks WHERE participant=? ORDER BY position',(p['id'],))]
    candidates=[x for x in tasks if x['status']=='returned']+[x for x in tasks if x['status'] in ('active','queued')]
    t=candidates[0] if candidates else None
    task=None
    if t:
        task={k:t[k] for k in ['id','arm','position','status','active','version']};task['case']=CASES[t['case_id']] if t['status']!='queued' else None
        previous=c.execute('SELECT a.answers,r.note FROM attempts a LEFT JOIN reviews r ON r.attempt_id=a.id WHERE a.task_id=? ORDER BY a.version DESC LIMIT 1',(t['id'],)).fetchone()
        if previous:task.update(saved_answers=json.loads(previous['answers']),review_note=previous['note'])
    return dict(participant=f'P{p["id"]:03d}',kind=p['kind'],role=p['role'],total=9,submitted=sum(x['status']=='submitted' for x in tasks),task=task,waiting_review=t is None)

def get_task(c,token,task_id):
    p=auth(c,token);t=c.execute('SELECT * FROM tasks WHERE id=? AND participant=?',(task_id,p['id'])).fetchone()
    if not t:raise ValueError('案件与当前参与者不匹配。')
    return t

def tick(c,t,delta):
    if t['status']!='active':return
    now=time.time();elapsed=max(0,now-(t['last_tick'] or now));add=min(max(0,float(delta)),elapsed+.25,30.)
    c.execute('UPDATE tasks SET active=active+?,last_tick=? WHERE id=?',(add,now,t['id']))

def grade(case_id,a):
    k=KEYS[case_id]
    def number(name,tol):
        try:return abs(float(a.get(name,''))-k[name])<=tol
        except (ValueError,TypeError):return False
    checks=[number('balance_delta',1),number('util_current',.1),number('util_reference',.1),a.get('payment_state')==k['payment_state'],a.get('action_evidence')==k['action_evidence']]
    refs=set(a.get('sources',[]));allowed={x['source_id'] for x in CASES[case_id]['history']}|{'R1','Q1','A1'}
    complete=len(str(a.get('note','')).strip())>=20 and set(k['source_ids']).issubset(refs) and refs.issubset(allowed) and a.get('next_step') in ['verify_action','verify_posting','handoff']
    return int(all(checks)),int(complete)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,obj,status=200):
        data=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        u=urlparse(self.path);q=parse_qs(u.query)
        try:
            if u.path=='/api/session':
                with connection() as c:self.send(session(c,q.get('token',[''])[0]))
                return
            if u.path=='/api/preview':
                case=q.get('case',['P19'])[0];arm=q.get('arm',['C'])[0]
                if case not in CASES or not CASES[case]['practice'] or arm not in 'ABC' or len(arm)!=1:raise ValueError('预览使用独立练习案件。')
                self.send(dict(preview=True,task=dict(id='preview',arm=arm,position=0,status='preview',active=0,version=0,case=CASES[case]),total=9,submitted=0));return
            if u.path=='/api/status':
                with connection() as c:
                    counts=dict(c.execute('SELECT kind,count(*) FROM participants GROUP BY kind').fetchall())
                self.send(dict(participants=counts,case_count=18,conditions=['A','B','C']));return
            paths={'/':'index.html','/app.js':'app.js','/style.css':'style.css','/font.ttf':'font.ttf'}
            if u.path not in paths:self.send({'error':'Not found'},404);return
            p=ROOT/paths[u.path];data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        except ValueError as e:self.send({'error':str(e)},400)
    def do_POST(self):
        try:
            if int(self.headers.get('Content-Length','0'))>100000:raise ValueError('请求过长。')
            a=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
            with connection() as c:
                if self.path=='/api/enroll':
                    if not a.get('consent'):raise ValueError('请先确认参与说明。')
                    role=a.get('role');experience=a.get('experience');kind=a.get('kind','participant')
                    if role not in ['风险复核','客户运营','模型策略','其他相关岗位'] or experience not in ['不足1年','1—3年','3年以上'] or kind not in ['participant','automation']:raise ValueError('请填写岗位与经验。')
                    # Block allocation separately by observed role/experience and
                    # kind; incomplete blocks are reported, never hidden.
                    seq=c.execute('SELECT count(*) FROM participants WHERE role=? AND experience=? AND kind=?',(role,experience,kind)).fetchone()[0]
                    token=secrets.token_urlsafe(24);pid=c.execute('INSERT INTO participants(token,role,experience,kind,created) VALUES(?,?,?,?,?)',(token,role,experience,kind,time.time())).lastrowid
                    for pos,(case,arm) in enumerate(assignment(seq)):
                        tid=secrets.token_hex(12);c.execute('INSERT INTO tasks(id,participant,case_id,arm,position,status) VALUES(?,?,?,?,?,?)',(tid,pid,case,arm,pos,'queued'))
                    self.send(dict(token=token,session=session(c,token)));return
                t=get_task(c,a.get('token',''),a.get('task_id',''))
                if self.path=='/api/start':
                    if t['status'] in ['queued','returned']:
                        now=time.time();c.execute('UPDATE tasks SET status=?,started=COALESCE(started,?),last_tick=? WHERE id=?',('active',now,now,t['id']))
                    self.send(session(c,a['token']));return
                if self.path=='/api/heartbeat':
                    tick(c,t,a.get('active_delta',0));self.send({'ok':True});return
                if self.path=='/api/submit':
                    prior=c.execute('SELECT id,task_id FROM attempts WHERE request_id=?',(a.get('request_id'),)).fetchone()
                    if prior:
                        if prior['task_id']!=t['id']:raise ValueError('重复请求标识不匹配。')
                        self.send(dict(ok=True,idempotent=True,session=session(c,a['token'])));return
                    if t['status']!='active':raise ValueError('请先开始任务。')
                    if not a.get('request_id'):raise ValueError('缺少提交标识。')
                    tick(c,t,a.get('active_delta',0));t=get_task(c,a['token'],t['id']);answer=a.get('answers',{})
                    if len(json.dumps(answer))>20000:raise ValueError('答题内容过长。')
                    obj,complete=grade(t['case_id'],answer);v=t['version']+1
                    c.execute('INSERT INTO attempts(task_id,version,request_id,answers,objective_pass,complete_fields,submitted,active_total,wall_total) VALUES(?,?,?,?,?,?,?,?,?)',(t['id'],v,a['request_id'],json.dumps(answer,ensure_ascii=False),obj,complete,time.time(),t['active'],time.time()-t['started']))
                    c.execute('UPDATE tasks SET status=?,version=? WHERE id=?',('submitted',v,t['id']))
                    self.send(dict(ok=True,session=session(c,a['token'])));return
                raise ValueError('未知操作。')
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as e:self.send({'error':str(e)},400)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8768);p.add_argument('--db',type=Path,default=DB);args=p.parse_args();DB=args.db.resolve();init()
    print(f'Trial workspace: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
