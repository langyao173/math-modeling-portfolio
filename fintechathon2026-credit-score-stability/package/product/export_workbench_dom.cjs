/* Export actual rendered DOM using the workbench's compact display mode. */
const fs=require('fs'),path=require('path');
const {JSDOM,VirtualConsole}=require(process.env.DOM_TEST_MODULE||'jsdom');
const root=__dirname,vc=new VirtualConsole();
const dom=new JSDOM(fs.readFileSync(path.join(root,'风险复核工作台.html'),'utf8'),{url:'https://workbench.test/',runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(w){const NativeDate=w.Date;w.Date=class extends NativeDate{constructor(...args){super(...(args.length?args:['2026-10-04T09:30:00Z']))}static now(){return new NativeDate('2026-10-04T09:30:00Z').valueOf()}};w.scrollTo=()=>{};w.fetch=async()=>{throw Error('Offline export')};w.structuredClone=structuredClone;w.crypto.randomUUID=()=>require('crypto').randomUUID();w.HTMLElement.prototype.scrollIntoView=function(){};w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};}});
(async()=>{
 const w=dom.window;await new Promise(r=>setTimeout(r,20));const c=w.CreditWorkbench;
 w.document.body.classList.add('showcase');w.eval('PAGE_SIZE=3');for(const id of ['quality-chart','attribution-chart'])w.document.getElementById(id).closest('.panel').classList.add('paper-secondary-metric');
 const r=c.data.cases.find(r=>r.capacity_guard?.guard_changed&&r.review_at10)||c.data.cases.find(r=>r.capacity_guard?.guard_changed);c.select(r.row_id,true);
 c.state.records={};c.state.events=[];c.state.drafts={};c.state.dirty=false;
 const demoRows=[r,...c.data.cases.filter(x=>x.review_at10&&x.row_id!==r.row_id).slice(0,2)];
 for(let i=0;i<demoRows.length;i++){
  const row=demoRows[i];
  if(!await c.commit([{case_key:row.case_key,expected_version:0,action:'claim',record:c.getRecord(row)}]))throw Error('claim failed');
  const rec=c.getRecord(row),pending=i!==1;
  const next={...rec,status:pending?'待补充':'核对中',due:'2026-10-06',topic:pending?'额度动作凭证':'账务与支付核对',note:pending?'待接续：核对额度调整发起方、审批依据与生效观察月。':'已读取近六期账务，继续核对最低应还与支付入账时点。'};
  if(!await c.commit([{case_key:row.case_key,expected_version:rec.version,action:'save',record:next}]))throw Error('save failed');
 }
 c.select(r.row_id,true);

 const out=path.join(root,'print_views');fs.mkdirSync(out,{recursive:true});
 for(const route of ['overview','queue','review','handling','agents','tasks']){
  if(route==='queue'){c.state.page=0;c.state.view='risk';} c.route(route);if(route==='agents')c.runLocalAgent('brief');
  for(const x of w.document.querySelectorAll('input'))x.setAttribute('value',x.value);
  for(const x of w.document.querySelectorAll('textarea'))x.textContent=x.value;
  for(const x of w.document.querySelectorAll('option'))x.toggleAttribute('selected',x.selected);
  let snapshot=dom.serialize().replace(/<script[\s\S]*?<\/script>/g,'');
  snapshot=snapshot.replace('</head>',`<style>
  @page{size:1220px 2400px;margin:0}body{width:1220px}body.showcase .sidebar{position:absolute;top:0;left:0;height:2400px;display:block!important}.sidebar-bottom,.sidebar-utility{display:none!important}nav button{display:block;width:100%;margin-bottom:5px;text-align:left}nav button svg{vertical-align:middle}.shell{margin-left:164px}.topbar{display:none!important}button,select,input,textarea{font-family:ReportCJK,Arial;line-height:1.5}#toast{display:none}dialog:not([open]){display:none}section{break-inside:avoid}.chart-svg{max-width:100%}.check-row{display:grid!important;grid-template-columns:minmax(0,1fr) 110px!important}.check-row input{grid-column:1/-1!important;width:100%;min-width:0}.form-grid{display:grid!important;grid-template-columns:1fr 1fr!important}.form-grid label{min-width:0}.form-grid input,.form-grid select{max-width:100%;width:100%;box-sizing:border-box;min-width:0}.wide-label textarea{width:100%;box-sizing:border-box}.protection-panel{padding:16px!important}.protection-panel .panel-head{margin-bottom:4px}.guard-chart{height:160px}.guard-chart svg{height:160px;width:100%}.guard-notes{font-size:12px}.dual-limit{margin-top:8px}.limit-item{padding:8px 12px}.limit-item p{margin:3px 0 0}.limit-track{margin:6px 0}.protection-grid{gap:12px}body[data-active-page=review] .evidence-grid{grid-template-columns:1.15fr 1fr}body[data-active-page=handling] .evidence-column{grid-template-columns:1fr 1fr!important}

  button{min-height:30px;height:auto!important;line-height:1.6!important;box-sizing:border-box}select,input{min-height:28px;height:auto;line-height:1.6!important}body .check-row{display:block!important;break-inside:avoid}body .check-row>div{display:inline-block!important;width:66%!important;vertical-align:middle}body .check-row>div strong,body .check-row>div small{display:block;line-height:1.6}body .check-row>select{display:inline-block!important;width:30%!important;vertical-align:middle}body .check-row>input{display:block!important;width:100%!important;margin-top:5px}body .signal-row{display:block!important;width:100%;text-align:left;min-height:72px;padding:10px 0!important}body .signal-row>div{display:block;line-height:1.6}body .signal-row>div b{float:right}body .signal-row p{display:block;line-height:1.6;margin-top:6px;margin-bottom:0}body .signal-track{display:block;height:3px;margin:6px 0}body .source-row,body .chain-step{display:block!important;width:100%;text-align:left;height:auto!important;min-height:54px;padding:8px 0!important}body .source-row>span,body .chain-step>span{display:inline-block;vertical-align:middle;line-height:1.6}body .source-row>span:nth-child(1),body .chain-step>span:nth-child(1){width:10%}body .source-row>span:nth-child(2),body .chain-step>span:nth-child(2){width:72%}body .source-row>span:nth-child(3),body .chain-step>span:nth-child(3){width:15%}body .source-row b,body .source-row small,body .chain-step b,body .chain-step small{display:block;line-height:1.65}body .page-heading .primary{padding:9px 14px!important}body .workflow-stepper{display:none!important}

  body .route-row{display:block!important;min-height:0!important;line-height:1.5!important;padding:7px 0!important}body .route-row-head{display:block;line-height:1.6}body .route-row-head strong{float:right}body .route-row p{display:block;margin-top:4px;line-height:1.5}body .case-progress{display:block;width:340px}body .case-progress>span{display:inline-block;white-space:nowrap;margin-right:10px}body .citation{display:inline!important;min-height:0!important;padding:0 3px!important;border:0!important;height:auto!important;line-height:1.6!important}details:not([open])>ol{display:none!important}

  body .risk-panel{position:relative}body #risk-distribution{position:absolute;left:16px;right:16px;top:90px;height:270px!important}body #risk-distribution svg{height:260px!important;width:100%;display:block}body .risk-panel>.legend{position:absolute;left:16px;bottom:20px}
  body[data-active-page=tasks] .task-board{grid-template-columns:repeat(5,minmax(0,1fr))}body[data-active-page=tasks] .task-card{font-size:11px;padding:8px}body[data-active-page=tasks] .task-card p{font-size:11px;line-height:1.5}.task-lane h2{font-size:12px}body[data-active-page=tasks] #event-list .event-row:nth-child(n+3){display:none}
  body[data-active-page=overview] .signal-row{min-height:55px!important;padding:7px 0!important}body[data-active-page=queue] .scatter-chart{height:180px!important}body[data-active-page=queue] .portfolio-grid .panel{padding:12px 14px!important}body[data-active-page=handling] .check-row{padding:8px 0!important}body[data-active-page=handling] .check-row>input{margin-top:3px!important}body[data-active-page=review] .paper-secondary-metric{display:none!important}
  </style></head>`);
  fs.writeFileSync(path.join(out,route+'.html'),snapshot);
 }
 fs.writeFileSync(path.join(out,'source.json'),JSON.stringify({type:'actual_DOM_print_export',browser_screenshot:false,exercise_clock:'2026-10-04T09:30:00Z',customer:r.customer_id,routes:6,case_records:'three local exercise records generated via claim/save; no measured human trial',exercise_cases:demoRows.map(x=>x.case_key)},null,2));w.close();console.log('Six actual DOM views exported.');
})().catch(e=>{console.error(e);process.exit(1)});
