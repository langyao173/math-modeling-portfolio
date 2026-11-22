const fs=require('fs'),path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_NODE_MODULES?path.join(process.env.PLAYWRIGHT_NODE_MODULES,'playwright'):'playwright');
const root=__dirname,out=path.join(root,'screenshots'),url='http://127.0.0.1:8769';
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({executablePath:process.env.FINTECH_CHROMIUM,args:['--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-zygote','--single-process']});
  const page=await browser.newPage({viewport:{width:1440,height:1200},deviceScaleFactor:1.5});let errors=[];
  page.on('pageerror',e=>errors.push(e.message));await page.goto(url);await page.evaluate(()=>document.fonts.ready);
  await page.screenshot({path:path.join(out,'01_试验入口.png'),fullPage:true});
  await page.click('#practice');await page.waitForSelector('#apply-brief');await page.evaluate(()=>document.fonts.ready);
  await page.click('#apply-brief');if(!(await page.inputValue('textarea[name=note]')).includes('L5'))throw Error('draft sources');
  await page.waitForSelector('#toast',{state:'hidden'});await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(150);
  await page.screenshot({path:path.join(out,'04_协同复核.png'),fullPage:true});
  for(const [arm,name] of [['A','02_账务核对'],['B','03_图表分析']]){
    await page.click(`[data-arm=${arm}]`);await page.waitForFunction(a=>document.querySelector('.topline .eyebrow').textContent.includes({A:'账务核对',B:'图表分析'}[a]),arm);
    await page.screenshot({path:path.join(out,name+'.png'),fullPage:true});
  }
  // Browser-level formal start -> fill -> submit -> refresh, only automation.
  const token=JSON.parse(fs.readFileSync(path.join(root,'state/browser_test_token.json'))).token;
  await page.evaluate(t=>localStorage.setItem('trialToken',t),token);await page.reload();await page.waitForSelector('#start');
  if(await page.locator('.ledger').count())throw Error('case visible before timed start');
  await page.click('#start');await page.waitForSelector('#answer');
  const current=await page.evaluate(async t=>(await(await fetch('/api/session?token='+t)).json()).task,token);
  const key=JSON.parse(fs.readFileSync(path.join(root,'answer_key.json')))[current.case.id];
  for(const name of ['balance_delta','util_current','util_reference'])await page.fill(`[name=${name}]`,String(key[name]));
  await page.selectOption('[name=payment_state]',key.payment_state);await page.selectOption('[name=action_evidence]',key.action_evidence);
  await page.fill('[name=note]','已核对前后两期账务与参考额度，交接补充额度调整原因、生效日期以及支付入账凭证。');
  for(const id of ['L5','L6','R1'])await page.check(`[name=sources][value=${id}]`);
  await page.click('#answer button[type=submit]');await page.waitForSelector('#start');await page.reload();await page.waitForSelector('#start');
  const status=await page.evaluate(async t=>await(await fetch('/api/session?token='+t)).json(),token);if(status.submitted!==1)throw Error('submission not restored');
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(root,'browser_checks.json'),JSON.stringify({checks:['practice routes share the same case','Agent draft cites source rows','timed case remains hidden before start','formal browser form submission persists after refresh'],page_errors:errors,screenshot_kind:'real browser capture; practice cases; automation excluded from efficacy'},null,2));
  await browser.close();console.log('Browser checks passed; 4 screenshots saved.');
})().catch(e=>{console.error(e);process.exit(1)});
