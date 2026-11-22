/* Capture real routes and exercise the same case APIs used by the visible forms. */
const fs=require('fs'),path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_NODE_MODULES?path.join(process.env.PLAYWRIGHT_NODE_MODULES,'playwright'):'playwright');
const ROOT=__dirname,OUT=path.join(ROOT,'screenshots');
const URL=process.env.FINTECH_WORKBENCH_URL||'http://127.0.0.1:8765';

(async()=>{
  fs.mkdirSync(OUT,{recursive:true});fs.mkdirSync(path.join(ROOT,'examples'),{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.FINTECH_CHROMIUM,args:['--no-sandbox','--disable-dev-shm-usage','--disable-gpu']});
  const context=await browser.newContext({viewport:{width:1220,height:1020},deviceScaleFactor:2,locale:'zh-CN'});
  const page=await context.newPage(),errors=[],checks=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(URL);await page.waitForFunction(()=>window.CreditWorkbench?.operationsReady);
  await page.evaluate(()=>document.fonts.ready);
  for(const route of ['overview','queue','review','handling','agents','tasks']){
    await page.click('nav [data-route="'+route+'"]');
    const active=await page.evaluate(()=>CreditWorkbench.state.route);
    if(active!==route)throw Error('Navigation failed: '+route+' / '+active);
  }
  checks.push('six distinct main routes');
  await page.click('nav [data-route="overview"]');
  await page.click('[data-route-scope="overdue"]');
  if(await page.evaluate(()=>CreditWorkbench.filtered().length)!==40)throw Error('Route drilldown');
  await page.click('nav [data-route="overview"]');
  const expected=await page.evaluate(()=>CreditWorkbench.data.cases.filter(r=>r.score_available&&r.risk_probability<.01).length);
  await page.click('[data-risk-bin="0"]');
  if(await page.evaluate(()=>CreditWorkbench.filtered().length)!==expected)throw Error('Histogram drilldown');
  await page.click('#clear-filters');await page.selectOption('#scope','risk');
  checks.push('route and histogram drilldowns match customer rows');

  const demoIds=await page.evaluate(()=>{
    const d=CreditWorkbench.data.cases;
    return [...d.filter(r=>r.review_at10).slice(0,5),...d.filter(r=>r.operating_queue==='差异核对').slice(0,1),...d.filter(r=>r.operating_queue==='逾期核对').slice(0,1)].map(r=>r.row_id);
  });
  for(let i=0;i<demoIds.length;i++){
    await page.evaluate(id=>CreditWorkbench.select(id,true),demoIds[i]);
    await page.click('#go-handle');
    const rec=await page.evaluate(()=>CreditWorkbench.getRecord(CreditWorkbench.data.cases.find(r=>r.row_id===CreditWorkbench.state.selected)));
    if(rec.version===0){
      await page.click('#claim-case');
      await page.waitForFunction(()=>CreditWorkbench.getRecord(CreditWorkbench.data.cases.find(r=>r.row_id===CreditWorkbench.state.selected)).version>=1);
      const ref=await page.evaluate(()=>{let r=CreditWorkbench.data.cases.find(r=>r.row_id===CreditWorkbench.state.selected);return {id:r.customer_key+' / '+r.account_key,month:r.anchor_month};});
      await page.selectOption('#check-identity','verified');await page.fill('#check-ref-identity',ref.id+' · M'+ref.month);
      await page.selectOption('#check-ledger','verified');await page.fill('#check-ref-ledger','M'+(ref.month-5)+' 至 M'+ref.month+' 已观察账务');
      await page.selectOption('#check-action',i%2===0?'needs':'pending');
      await page.selectOption('#review-status',i%2===0?'待补充':'核对中');
      await page.selectOption('#review-owner',i%3===0?'风险复核岗':i%3===1?'客户服务岗':'策略运营岗');
      await page.selectOption('#review-anchor_policy','handoff');
      await page.fill('#review-due','2026-10-0'+(5+i%4));
      await page.fill('#review-note',i%2===0?'已核对客户、账户与月度账务；接续补充额度动作凭证，确认生效时间及参考周期处理。':'已定位近期还款与最低应还记录；接续核对账务时点和额度调整依据。');
      await page.selectOption('#review-result','需补充动作依据');
      await page.click('#save-review');
      await page.waitForFunction(()=>!CreditWorkbench.state.dirty&&CreditWorkbench.getRecord(CreditWorkbench.data.cases.find(r=>r.row_id===CreditWorkbench.state.selected)).version>=2);
    }
  }
  checks.push('seven actual claim/save workflows with evidence references');
  await page.evaluate(id=>CreditWorkbench.select(id,true),demoIds[0]);
  await page.click('#open-copilot');await page.click('[data-agent="utilization"]');
  await page.click('#apply-brief');
  if(await page.evaluate(()=>CreditWorkbench.state.route)!=='handling')throw Error('Draft handoff route');
  if(!(await page.inputValue('#review-note')).includes('使用率'))throw Error('Draft handoff data');
  await page.click('#save-review');await page.waitForFunction(()=>!CreditWorkbench.state.dirty);
  const activeCase=await page.evaluate(()=>CreditWorkbench.state.selected);
  const version=await page.evaluate(()=>CreditWorkbench.getRecord(CreditWorkbench.data.cases.find(r=>r.row_id===CreditWorkbench.state.selected)).version);
  await page.reload();await page.waitForFunction(()=>CreditWorkbench.operationsReady);
  await page.evaluate(id=>CreditWorkbench.select(id,true),activeCase);
  const restored=await page.evaluate(()=>CreditWorkbench.getRecord(CreditWorkbench.data.cases.find(r=>r.row_id===CreditWorkbench.state.selected)).version);
  if(restored!==version)throw Error('Case persistence');
  checks.push('Agent draft -> editable handling form -> persisted same case');
  await page.click('#open-copilot');await page.click('[data-agent="utilization"]');
  for(const ref of ['S2','S3','S4','S5','S6']){
    await page.click('.source-register [data-ref="'+ref+'"]');
    if(!(await page.locator('#dialog-body').innerText()).trim())throw Error('Empty evidence '+ref);
    await page.click('#dialog-close');
  }
  checks.push('five evidence references open actual case content');
  const snapshot=await page.evaluate(()=>({...CreditWorkbench.exportObject(),workspace_context:'公开历史账务的操作演示；案件状态由本次领取、核对和保存操作生成。'}));
  fs.writeFileSync(path.join(ROOT,'examples','展示案件.json'),JSON.stringify(snapshot,null,2));
  await page.click('#presentation-toggle');
  const routes=[['overview','01_客群总览'],['queue','02_风险队列'],['review','03_客户分析'],['handling','04_证据办理'],['agents','05_AI协同'],['tasks','06_任务接续']];
  const capture=[];
  for(const [route,name] of routes){
    await page.click('nav [data-route="'+route+'"]');
    if(route==='queue'){await page.selectOption('#scope','risk');await page.click('#clear-filters');}
    if(route==='agents')await page.click('[data-agent="utilization"]');
    await page.evaluate(()=>document.fonts.ready);
    if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw Error('Horizontal overflow '+route);
    const screenshot=path.join(OUT,name+'.png');await page.screenshot({path:screenshot,fullPage:true});
    const box=await page.evaluate(()=>({width:innerWidth,height:Math.max(innerHeight,document.documentElement.scrollHeight)}));
    capture.push({route,name,screenshot:'screenshots/'+name+'.png',viewport:box,device_scale:2,selected_customer:await page.evaluate(()=>CreditWorkbench.state.selected)});
  }
  const routeCounts=await page.evaluate(()=>CreditWorkbench.data.cases.reduce((s,r)=>(s[r.operating_queue]=(s[r.operating_queue]||0)+1,s),{}));
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(ROOT,'page_verification.json'),JSON.stringify({checks,page_errors:errors,route_counts:routeCounts,persisted_case_version:restored,workflow_records:Object.keys(snapshot.records).length,saved_events:snapshot.events.length},null,2));
  fs.writeFileSync(path.join(ROOT,'page_capture.json'),JSON.stringify({source:'风险复核工作台.html',capture_method:'actual interactive browser routes',layout:'built-in 展示视图',pages:capture},null,2));
  console.log(JSON.stringify({checks,pages:capture.map(c=>({page:c.name,...c.viewport})),errors},null,2));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
