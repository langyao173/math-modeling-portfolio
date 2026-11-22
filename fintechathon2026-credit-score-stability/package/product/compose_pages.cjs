/* Assemble unchanged browser screenshots into aligned publication panels. */
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const {chromium}=require(process.env.PLAYWRIGHT_NODE_MODULES?path.join(process.env.PLAYWRIGHT_NODE_MODULES,'playwright'):'playwright');
const ROOT=__dirname;
const pages=JSON.parse(fs.readFileSync(path.join(ROOT,'page_capture.json'),'utf8')).pages;
const labels={overview:['客群总览','作业分流与评分分布'],queue:['风险队列','筛查、分派与按序复核'],review:['客户分析','账务轨迹与同尺度解释'],handling:['证据办理','核对凭证、责任与跟进'],agents:['AI 协同','来源引用与办件草稿'],tasks:['任务接续','跨岗进度与客户时间线']};

function html(routes){
  return `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>043 · 工作台页面</title><style>
  @font-face{font-family:ReportCJK;src:url('../assets/fonts/ReportChartCJK.ttf')}*{box-sizing:border-box}body{margin:0;background:#fff;color:#304737;font-family:ReportCJK,sans-serif}.figure{width:2440px;padding:12px 18px 18px}.pages{display:grid;grid-template-columns:1fr 1fr;gap:34px 30px}.panel{min-width:0}.caption{display:flex;align-items:baseline;gap:12px;height:76px;padding:5px 1px 10px;font-size:40px}.caption b{font-family:Arial,sans-serif;font-size:44px}.caption span{font-size:31px;color:#73806d;margin-left:7px}.screen{border:1px solid #cbd5c4;background:#f4f6f3;height:1120px;overflow:hidden}.screen img{display:block;width:100%;height:100%;object-fit:contain;object-position:top left}
  </style><div class="figure"><div class="pages">${routes.map((route,i)=>{
    const p=pages.find(p=>p.route===route),[title,description]=labels[route];
    return `<section class="panel"><div class="caption"><b>${String.fromCharCode(97+i)}</b>${title}<span>${description}</span></div><div class="screen"><img src="${p.screenshot}" alt="${title}实际页面"></div></section>`;
  }).join('')}</div></div></html>`;
}

(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.FINTECH_CHROMIUM,args:['--no-sandbox','--disable-dev-shm-usage','--disable-gpu']});
  const outputs=[];
  for(const [stem,routes] of [['workbench_paper',['overview','review','handling','agents']],['workbench_six',['overview','queue','review','handling','agents','tasks']]]){
    const file=path.join(ROOT,stem+'.html');fs.writeFileSync(file,html(routes));
    const page=await browser.newPage({viewport:{width:2440,height:1400},deviceScaleFactor:2});
    await page.goto(pathToFileURL(file).href);await page.evaluate(()=>document.fonts.ready);
    await page.waitForFunction(()=>Array.from(document.images).every(x=>x.complete&&x.naturalWidth>0));
    const image=path.join(ROOT,stem+'.png');await page.locator('.figure').screenshot({path:image});
    outputs.push({file:stem+'.png',layout:'two columns',source_routes:routes,panels:routes.length,composition:'unchanged full-page screenshots; aligned labels and spacing'});
    await page.close();
  }
  fs.writeFileSync(path.join(ROOT,'screenshot_provenance.json'),JSON.stringify({source:'风险复核工作台.html',source_pages:'page_capture.json',outputs},null,2));
  console.log(JSON.stringify(outputs,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
