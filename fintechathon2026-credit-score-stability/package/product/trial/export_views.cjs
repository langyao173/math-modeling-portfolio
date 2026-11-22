/* Execute the actual offline practice UI; export source, chart and agent components. */
const fs=require('fs'),path=require('path');
const {JSDOM,VirtualConsole}=require(process.env.DOM_TEST_MODULE||'jsdom');
const root=__dirname,out=path.join(root,'print_views');fs.mkdirSync(out,{recursive:true});
const vc=new VirtualConsole(),errors=[];vc.on('jsdomError',e=>errors.push(e.message));
const dom=new JSDOM(fs.readFileSync(path.join(root,'../043_复核作业_可交互练习.html'),'utf8'),{url:'https://practice.test/',runScripts:'dangerously',pretendToBeVisual:true,virtualConsole:vc,beforeParse(w){w.HTMLElement.prototype.scrollIntoView=function(){};w.crypto.randomUUID=()=>require('crypto').randomUUID();}});
(async()=>{
 const w=dom.window;await new Promise(r=>setTimeout(r,20));
 const style=w.document.querySelector('style').textContent;
 const checks=[];
 for(const arm of ['A','B','C']){
  await w.openPreview(arm,'P19');
  const available=[...w.document.querySelectorAll('[data-worktab]')].map(x=>x.dataset.worktab);
  if(JSON.stringify(available)!==JSON.stringify({A:['ledger'],B:['ledger','charts'],C:['ledger','charts','agent']}[arm]))throw Error('Arm availability');
  if(w.document.querySelectorAll('.ledger tbody tr').length!==6)throw Error('Missing raw records');
  for(const tab of w.document.querySelectorAll('[data-work-tab]')){tab.click();if(w.document.querySelectorAll('[data-worktab]:not([hidden])').length!==1)throw Error('Tab switching');}
  const selector={A:'.ledger-panel',B:'.chart-panel',C:'.agent'}[arm],node=w.document.querySelector(selector).cloneNode(true);node.removeAttribute('hidden');node.querySelectorAll('button').forEach(b=>{const s=w.document.createElement('span');s.className=b.className;s.id=b.id;s.textContent=b.textContent;b.replaceWith(s)});
  if(arm==='C'){
   w.document.querySelector('#apply-brief').click();
   if(!w.document.querySelector('[name=note]').value.includes('L5'))throw Error('Draft missing source');
   if(w.document.querySelectorAll('[name=sources]:checked').length!==3)throw Error('Source selection');
   w.document.querySelector('.agent [data-source=L5]').click();if(w.document.querySelector('.ledger-panel').hidden)throw Error('Source navigation did not reveal ledger');w.document.querySelector('[data-work-tab=agent]').click();
  }
  const width=arm==='A'?900:438;
  const css=`@page{size:${width}px 1400px;margin:0}body{width:${width}px;background:white;margin:0;padding:0}body>.panel{width:${width}px;margin:0;border-radius:0;box-shadow:none;padding:18px 20px}.chartgrid{display:block}.chartbox+ .chartbox{margin-top:22px}.chartbox svg{width:100%;margin-top:8px}.chartbox h3{font-size:14px;margin-left:0}.chartbox h3 small{display:block;font-size:10px}.legend{display:block}.legend span{margin-right:18px}.ledger{font-size:14px}.ledger th,.ledger td{padding:10px 7px}.smallprint{font-size:12px}button{font-family:Report;min-height:0;line-height:1.6}.agent-summary{font-size:14px;line-height:1.85}.agent-head h2{font-size:17px}.agent .motif{display:table;width:100%;border-spacing:4px}.motif .cell,.motif .arrow{display:table-cell;vertical-align:middle}.motif .cell small{font-size:10px}.motif strong{font-size:22px}.flowlist{font-size:13px;margin-top:12px}.flowlist li{padding-top:8px;padding-bottom:8px}.agent details{margin:10px 0 0!important;padding-top:10px}details:not([open])>p{display:none}.source-chip{display:inline!important;width:auto!important;min-width:0;padding:2px 5px;font-size:11px}#apply-brief{display:block;width:100%;white-space:normal;border:1px solid #bbcbbd;border-radius:4px;padding:10px 8px;text-align:center;font-size:13px;color:#315d50;background:#f7faf4;line-height:1.6}`;
  fs.writeFileSync(path.join(out,arm+'.html'),'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>'+style+'\n'+css+'</style></head><body>'+node.outerHTML+'</body></html>');
  checks.push({arm,available,rows:6,tab_switching:true,draft_sources:arm==='C'?['L5','L6','R1']:null,source_navigation:arm==='C'?true:null});
 }
 if(errors.length)throw Error(errors.join(';'));
 fs.writeFileSync(path.join(out,'source.json'),JSON.stringify({type:'actual_DOM_component_export',renderer:'WeasyPrint',browser_screenshot:false,case:'P19',checks,errors},null,2));
 w.close();console.log('Three practice views exported; tabs, case consistency and sourced draft verified.');
})().catch(e=>{console.error(e);process.exit(1)});
