"""Build the standalone practice page from the same live trial UI and three practice cases."""
from pathlib import Path
import base64,json
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'product/trial'
cases={c['id']:c for c in json.loads((T/'cases.json').read_text()) if c['practice']}
assert set(cases)=={'P19','P20','P21'}
bootstrap='const OFFLINE_CASES='+json.dumps(cases,ensure_ascii=False)+';\n'+r'''window.fetch=async(url)=>{const q=new URL(url,'http://offline');if(q.pathname==='/api/preview'){const arm=q.searchParams.get('arm'),id=q.searchParams.get('case');if(!OFFLINE_CASES[id]||!['A','B','C'].includes(arm))return {ok:false,json:async()=>({error:'请选择练习案件'})};return {ok:true,json:async()=>({preview:true,task:{id:'preview',arm,position:0,status:'preview',active:0,version:0,case:OFFLINE_CASES[id]},total:9,submitted:0})};}return {ok:false,json:async()=>({error:'正式登记请启动复核作业服务；本页用于独立练习。'})};};'''
app=(T/'app.js').read_text().replace("if(token)resume();else landing();","token=null;openPreview('C','P19');")
css=(T/'style.css').read_text().replace("url('/font.ttf')","url('data:font/ttf;base64,"+base64.b64encode((T/'font.ttf').read_bytes()).decode()+"')")
html=(T/'index.html').read_text().replace('<link rel="stylesheet" href="/style.css">','<style>'+css+'</style>').replace('<script src="/app.js"></script>','<script>'+bootstrap+'\n'+app+'</script>')
(ROOT/'product/043_复核作业_可交互练习.html').write_text(html)
print('Standalone practice page built from shared UI source.')
