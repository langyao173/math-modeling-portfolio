/* Execute the self-contained artifact in a DOM test environment; no screenshot claim. */
const fs=require('fs'),path=require('path');
const {JSDOM,VirtualConsole}=require(process.env.DOM_TEST_MODULE||'jsdom');
const root=__dirname,errors=[],vc=new VirtualConsole();
vc.on('jsdomError',e=>errors.push(e.message));
const html=fs.readFileSync(path.join(root,'风险复核工作台.html'),'utf8');
const dom=new JSDOM(html,{url:'https://workbench.test/',runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(w){w.scrollTo=()=>{};w.fetch=async()=>{throw Error('Offline DOM test')};w.structuredClone=structuredClone;w.crypto.randomUUID=()=>require('crypto').randomUUID();w.HTMLElement.prototype.scrollIntoView=function(){};w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};}});
(async()=>{
 const w=dom.window;await new Promise(r=>setTimeout(r,20));
 const c=w.CreditWorkbench;if(!c)throw Error(errors.join('\n')||'Workbench not initialized');
 const assert=(b,m)=>{if(!b)throw Error(m)};
 for(const route of ['overview','queue','review','handling','agents','tasks']){c.route(route);assert(c.state.route===route,'route '+route)}
 c.route('queue');w.document.querySelector('#open-boundary').click();assert(c.filtered().length===2,'Boundary filter');
 const r=c.filtered()[0];c.select(r.row_id,true);assert(w.document.querySelector('.protection-panel').textContent.includes('已执行边界保护'),'Protection details');
 c.source('S4');assert(w.document.querySelector('#dialog-body').textContent.includes('必要放宽'),'Receipt source');w.document.querySelector('#dialog-close').click();
 c.route('handling');assert(w.document.querySelector('#review-action_initiator'),'Action evidence form');
 const before=c.getRecord(r);await c.commit([{case_key:r.case_key,expected_version:before.version,action:'claim',record:before}]);
 assert(c.getRecord(r).assignee,'Claim case');
 const actor=w.document.querySelector('#review-action_initiator');actor.value='bank';w.document.querySelector('#review-qualification_change').value='unchanged';w.document.querySelector('#review-action_evidence_id').value='AUTOMATED-TEST';w.document.querySelector('#review-action_evidence_month').value=String(r.anchor_month);w.document.querySelector('#review-note').value='软件验证记录';w.document.querySelector('#review-receipt').value='AUTOMATED-TEST';
 const draft=c.readForm();assert(draft.action_initiator==='bank'&&draft.action_evidence_id==='AUTOMATED-TEST','Read extended form');
 await c.saveReview();assert(c.getRecord(r).action_evidence_id==='AUTOMATED-TEST','Save extended evidence');
 const tool=c.toolRegistry.explain_reference(r);assert(tool.capacity_receipt.guard_changed,'Agent protected receipt');
 c.route('agents');c.runLocalAgent('readiness');assert(w.document.querySelector('#agent-pane').textContent.length>50,'Agent output');
 const risks=c.data.cases.filter(r=>r.review_at10);assert(risks.length===268&&risks.every((r,i)=>r.risk_rank===i+1),'Final risk ordering');
 const legitErrors=errors.filter(x=>!x.startsWith('Could not parse CSS stylesheet'));
 assert(!legitErrors.length,legitErrors.join('\n'));
 const summary={six_routes:true,boundary_filter_count:2,protected_list_count:268,ordered_by_final_q:true,action_fields_read_and_saved:true,agent_reads_protection_receipt:true,measurement_type:'jsdom_software_execution',visual_browser_capture:false,errors:legitErrors};
 fs.writeFileSync(path.join(root,'dom_verification.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));
 // Export the actual page DOM for a separate, explicitly labelled print preview.
 c.state.records={};c.state.events=[];c.state.drafts={};c.state.dirty=false;
 const printDir=path.join(root,'print_views');fs.mkdirSync(printDir,{recursive:true});
 for(const route of ['overview','queue','review','handling','agents','tasks']){
  if(route==='queue'){c.state.view='risk';c.state.workView='all';c.state.clue='';c.state.search='';}
  c.route(route);if(route==='agents')c.runLocalAgent('readiness');
  for(const x of w.document.querySelectorAll('input'))x.setAttribute('value',x.value);
  for(const x of w.document.querySelectorAll('textarea'))x.textContent=x.value;
  for(const x of w.document.querySelectorAll('option'))x.toggleAttribute('selected',x.selected);
  let snapshot=dom.serialize().replace(/<script[\s\S]*?<\/script>/g,'');
  snapshot=snapshot.replace('</head>','<style>@page{size:1220px 2200px;margin:0}body{width:1220px}nav button{display:block;width:100%}.sidebar{position:absolute;top:0;left:0;height:1200px;display:block!important}.sidebar-bottom{display:none!important}.topbar{display:none!important}button,select,input,textarea{font-family:ReportCJK,Arial;min-height:28px;line-height:1.5}nav button{margin-bottom:6px;text-align:left}.topbar{position:static}.shell{margin-left:194px}.page{break-inside:avoid}#toast{display:none}dialog:not([open]){display:none}.chart-svg{max-width:100%}.check-row{display:block!important}.check-row>div{width:100%!important;margin-bottom:6px}.check-row input,.check-row select{display:block;width:100%!important;max-width:100%!important;margin:5px 0;box-sizing:border-box}.form-grid{display:block!important}.form-grid>label{display:block;margin-bottom:8px}.review-layout{grid-template-columns:minmax(0,1fr) 300px!important}</style></head>');
  fs.writeFileSync(path.join(printDir,route+'.html'),snapshot);
 }
 w.close();
})().catch(e=>{console.error(e);process.exit(1)});
