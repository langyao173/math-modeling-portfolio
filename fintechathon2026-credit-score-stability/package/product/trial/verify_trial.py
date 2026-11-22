"""End-to-end checks against a separate automation database; no human efficacy."""
from pathlib import Path
import collections,csv,importlib.util,json,sqlite3,urllib.request,uuid
ROOT=Path(__file__).resolve().parent;URL='http://127.0.0.1:8769';DB=ROOT/'state/automation.sqlite'
def call(path,data=None):
    req=urllib.request.Request(URL+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req) as r:return json.load(r)

def main():
    checks=[];tokens=[]
    for _ in range(12):
        s=call('/api/enroll',dict(role='风险复核',experience='3年以上',kind='automation',consent=True));tokens.append(s['token'])
        assert s['session']['task']['case'] is None
    checks.append('正式案件开始前不显示内容；程序登记全部标记 automation')
    with sqlite3.connect(DB) as c:
        ids=c.execute("SELECT id FROM participants WHERE kind='automation' ORDER BY id DESC LIMIT 12").fetchall();rows=[]
        for (pid,) in ids:
            rr=c.execute('SELECT case_id,arm FROM tasks WHERE participant=?',(pid,)).fetchall();assert len(rr)==9 and len({x[0] for x in rr})==9 and collections.Counter(x[1] for x in rr)==dict(A=3,B=3,C=3);rows+=rr
        counts=collections.Counter(rows);assert len(counts)==54 and set(counts.values())=={2}
    checks.append('12 人完整区组中，18 案件 × 3 界面各出现 2 次；每人 9 案无重复')
    token=tokens[0];s=call('/api/session?token='+token);tid=s['task']['id'];s=call('/api/start',dict(token=token,task_id=tid))
    case=s['task']['case'];flat=json.dumps(case).lower();assert 'outcome' not in flat and 'target' not in flat and 'customer_id' not in flat
    key=json.loads((ROOT/'answer_key.json').read_text())[case['id']];answers={**key,'sources':key['source_ids'],'next_step':'verify_action','note':'已按前后两期账务与参考额度核对，并接续补充额度调整依据和支付入账时间。'};answers.pop('source_ids')
    request=dict(token=token,task_id=tid,active_delta=999,request_id=str(uuid.uuid4()),answers=answers)
    call('/api/submit',request);repeat=call('/api/submit',request);assert repeat['idempotent']
    checks.append('相同请求重复提交仅保留一条版本记录；数值与引用正确性在服务端判断')
    with sqlite3.connect(DB) as c:
        aid,ver,obj,complete,active=c.execute('SELECT id,version,objective_pass,complete_fields,active_total FROM attempts WHERE task_id=?',(tid,)).fetchone();assert ver==1 and obj==1 and complete==1 and active<2
        c.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?)',(aid,1,1,'return','请明确额度凭证接续事项。','程序校验',0));c.execute("UPDATE tasks SET status='returned' WHERE id=?",(tid,));c.commit()
    resumed=call('/api/session?token='+token);assert resumed['task']['id']==tid and resumed['task']['status']=='returned' and resumed['task']['saved_answers']['note']==answers['note']
    call('/api/start',dict(token=token,task_id=tid));request['request_id']=str(uuid.uuid4());request['active_delta']=0;call('/api/submit',request)
    with sqlite3.connect(DB) as c:
        assert c.execute('SELECT count(*) FROM attempts WHERE task_id=?',(tid,)).fetchone()[0]==2
    checks.append('刷新恢复已填材料；退回同案接续，产生第二版本并累计作业时间')
    spec=importlib.util.spec_from_file_location('trial_server',ROOT/'server.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bad={**answers,'util_reference':99999};assert module.grade(case['id'],bad)[0]==0
    bad={**answers,'sources':['X99']};assert module.grade(case['id'],bad)[1]==0
    checks.append('错误读数与无效证据引用均不能获得完整正确判定')
    for endpoint in ['/answer_key.json','/cases.json','/state/automation.sqlite']:
        try:call(endpoint);raise AssertionError(endpoint)
        except urllib.error.HTTPError as ex:assert ex.code==404
    for arm in 'ABC':
        p=call('/api/preview?arm='+arm+'&case=P19');assert p['task']['case']['id']=='P19'
    try:call('/api/preview?arm=A&case=T01');raise AssertionError('formal preview exposed')
    except urllib.error.HTTPError as ex:assert ex.code==400
    checks.append('答案、数据库与完整案件库不通过网页公开；练习不能读取正式案件')
    (ROOT/'functional_checks.json').write_text(json.dumps(dict(checks=checks,real_participants=0,efficacy_measured=False,automation_database='state/automation.sqlite'),ensure_ascii=False,indent=2))
    (ROOT/'state/browser_test_token.json').write_text(json.dumps(dict(token=tokens[-1])))
    print(json.dumps(dict(checks=checks,passed=len(checks)),ensure_ascii=False))

if __name__=='__main__':main()
