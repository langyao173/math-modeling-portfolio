/* Six operational routes share the same customer, evidence and case state. */
const routeSpecs=[
  ['risk','名单复核','按统一评分领取、核对与复审','#567c68'],
  ['shadow','差异核对','核对更新差异，接续风险线索','#b99a60'],
  ['overdue','逾期核对','核对应还、入账与逾期天数','#bd8c75'],
  ['missing','账务补齐','补齐当前观察期账务','#9a93a5'],
  ['observe','持续观察','保存快照，等待新信息','#a9b7a3']
];
const scoreBins=[0,.01,.025,.05,.1,.2,1.000001];
const scoreBinLabels=['0–1','1–2.5','2.5–5','5–10','10–20','≥20'];
state.riskBand=null;
const priorFiltered=filtered;
filtered=function(){return priorFiltered().filter(r=>state.riskBand===null||
  (r.score_available&&r.risk_probability>=scoreBins[state.riskBand]&&r.risk_probability<scoreBins[state.riskBand+1]));};

function riskHistogram(){
  const bins=scoreBinLabels.map((label,i)=>{
    const rows=ROWS.filter(r=>r.score_available&&r.risk_probability>=scoreBins[i]&&r.risk_probability<scoreBins[i+1]);
    return {label,count:rows.length,selected:rows.filter(r=>r.review_at10).length};
  });
  const mx=Math.max(1,...bins.map(b=>b.count)),w=560,h=235,left=38,bottom=188;
  let s='';
  for(let j=0;j<=2;j++){
    const v=Math.ceil(mx/100)*100*j/2,y=bottom-v/(Math.ceil(mx/100)*100)*147;
    s+=line(left,y,549,y,'#e9eee8')+text(left-9,y+4,fmt(v,0),'text-anchor="end" style="font-size:12px"');
  }
  const scale=147/(Math.ceil(mx/100)*100);
  bins.forEach((b,i)=>{
    const x=53+i*82,base=bottom-b.count*scale,selected=b.selected*scale;
    s+=`<g data-risk-bin="${i}" tabindex="0" role="button" aria-label="${b.label}%评分，${b.count}户"><rect x="${x}" y="${base}" width="48" height="${Math.max(0,b.count*scale)}" rx="2" fill="#c3cec2"/><rect x="${x}" y="${bottom-selected}" width="48" height="${Math.max(0,selected)}" fill="#567c68"/>`;
    s+=text(x+24,base-9,String(b.count),'text-anchor="middle" style="fill:#435e50;font-size:13px"');
    s+=text(x+24,bottom+22,b.label,'text-anchor="middle" style="font-size:12px"')+'</g>';
  });
  s+=text(548,231,'q（%）','text-anchor="end" style="font-size:12px"');
  return svg(s,w,h,'统一风险评分分组人数，深绿为进入复核名单的客户');
}

function renderOverview(){
  $('#overview-period').value=state.period;
  $('#route-distribution').innerHTML=routeSpecs.map(([key,name,description,color])=>{
    const n=ROWS.filter(r=>r.operating_queue===name).length;
    return `<button class="route-row" data-route-scope="${key}"><div class="route-row-head"><span><i style="background:${color}"></i>${name}</span><strong>${fmt(n,0)} <small>户 · ${fmt(100*n/ROWS.length,2)}%</small></strong></div><div class="route-track"><span style="width:${100*n/ROWS.length}%;background:${color}"></span></div><p>${description}<span>查看 →</span></p></button>`;
  }).join('');
  $('#risk-distribution').innerHTML=riskHistogram();
  $('#overview-scatter').innerHTML=scatterChart(ROWS);
  $('#overview-scatter-label').textContent=periodName()+' · 全量可配对账务';
  $('#overview-signals').innerHTML=Object.entries(clues).map(([key,c])=>{
    const n=ROWS.filter(c.test).length;
    return `<button class="signal-row" data-overview-clue="${key}"><div><span>${c.name}</span><b>${fmt(n,0)} 户</b></div><span class="signal-track"><i style="width:${100*n/ROWS.length}%"></i></span><p>${{opposite:'核对余额增长与额度调整的不同来源',shortfall:'核对还款时点与最低应还',dpd:'查看历史逾期与当前账务',quality:'展开统一评分的更新依据'}[key]}<span>筛查 →</span></p></button>`;
  }).join('');
}

function renderAgentCenter(){
  const r=current(),rec=getRec(r),draft=state.drafts[r.case_key];
  $('#agent-context').innerHTML=`<span>当前客户 <strong>${esc(r.customer_key)}</strong></span><span>观察月 <strong>M${r.anchor_month}</strong></span><span>案件 <strong>v${rec.version} · ${esc(rec.status)}</strong></span><span>统一风险 q <strong>${r.risk_probability===null?'—':fmt(r.risk_probability*100)+'%'}</strong></span>`;
  const stages=[
    ['01','账务检索','账户身份 · 近六期账务','S2'],
    ['02','数值分析','使用率三视图 · 评分更新','S3'],
    ['03','材料检查','四项证据 · 待补事项','S5'],
    ['04','办件草稿','带来源简报 · 接续清单','S6']
  ];
  $('#agent-chain').innerHTML=stages.map(([no,title,description,ref])=>`<button class="chain-step ${draft?'complete':''}" data-ref="${ref}"><span class="chain-no">${no}</span><span><b>${title}</b><small>${description}</small></span><span class="chain-state">${draft?'已完成':'待运行'}</span></button>`).join('');
  const sources=[['S1','账户与评分快照',r.customer_key+' · M'+r.anchor_month],['S2','原始账务明细',r.account_history.filter(h=>h.balance!==null).length+'期已观察账务'],['S3','使用率计算',periodName()+' · 三种读数'],['S4','统一评分更新','参考额度 '+amount(r.reference_limit)],['S5','核查清单',Object.values(rec.checks).filter(c=>c.status==='verified'&&c.reference).length+'/4 项证据已核对'],['S6','案件接续',rec.owner+' · '+(rec.assignee||'待领取')]];
  $('#agent-source-register').innerHTML=sources.map(([ref,title,detail])=>`<button class="source-row" data-ref="${ref}"><span class="source-ref">${ref}</span><span><b>${title}</b><small>${esc(detail)}</small></span><span>↗</span></button>`).join('');
  $('#source-register-count').textContent=sources.length+' 项来源';
  renderAgentRuns();
  $('#agent-runs').querySelectorAll('.run-row').forEach((row,i)=>{if(i>2)row.hidden=true;});
}

// Keep one evidence form and one copilot instance so values and events stay aligned.
$('#agent-workspace-output').append($('#agent-pane'));
const sixPageSource=source;
source=function(id){
  const r=current(),c=getRec(r);
  if(id==='S5'){
    dialog('本案核查清单',`<p>${esc(r.customer_key)} · 案件 v${c.version}</p><table><thead><tr><th>核对项目</th><th>进度</th><th>已登记依据</th></tr></thead><tbody>${Object.entries(WF.checkNames).map(([key,label])=>`<tr><td>${label}</td><td>${{verified:'已核对',needs:'待补材料',pending:'待核对'}[c.checks[key].status]}</td><td>${esc(c.checks[key].reference||'待补充')}</td></tr>`).join('')}</tbody></table><p>责任岗：${esc(c.owner)}；经办人：${esc(c.assignee||'待领取')}。</p>`);return;
  }
  if(id==='S6'){
    dialog('案件接续记录',`<p>${esc(r.customer_key)} · ${esc(c.status)} · v${c.version}</p><p>${esc(c.note||'领取案件后，可记录核对依据与接续事项。')}</p>${eventHTML(state.events.filter(e=>e.case_key===r.case_key))}`);return;
  }
  return sixPageSource(id);
};
renderAgentRuns=function(){
  const labels={brief:'复核简报',utilization:'使用率说明',quality:'评分更新说明',handoff:'交接说明',readiness:'材料检查'};
  const tools={lookup_case:'案件定位',retrieve_history:'账务检索',calculate_utilization:'使用率计算',explain_reference:'评分分解',draft_review_plan:'核查清单',prepare_handoff:'接续记录',language_synthesis:'文字整理'};
  $('#agent-runs').innerHTML=state.agentRuns.slice(-3).reverse().map(x=>{
    const r=ROWS.find(r=>r.case_key===x.case_key);
    return `<div class="run-row"><span>${esc(r?.customer_key||'当前案件')} · ${labels[x.intent]||'材料整理'}<small>${x.tools.map(t=>tools[t]||t).join(' → ')}</small></span><small>${new Date(x.created_at).toLocaleTimeString('zh-CN')}</small></div>`;
  }).join('')||'<div class="empty">选择一项复核任务，运行记录随草稿保存。</div>';
};
const sixPageRenderAgent=renderAgent;
renderAgent=function(){sixPageRenderAgent();renderAgentCenter();};
const sixPageQueue=renderQueue;
renderQueue=function(){
  sixPageQueue();
  if(state.riskBand!==null)$('#queue-context').textContent+=' / q '+scoreBinLabels[state.riskBand]+'%';
};
const sixPageReview=renderReview;
renderReview=function(){
  sixPageReview();
  const r=current(),handling=state.route==='handling';
  $('#customer-title').textContent=handling?'证据办理':'客户分析';
  $('#case-eyebrow').textContent=handling?'CASE HANDLING':'CUSTOMER ANALYSIS';
  $('#account-meta').textContent=`${r.customer_key} · ${r.account_key} · 观察月 M${r.anchor_month}`;
  $('#customer-ribbon').querySelector('span').innerHTML=`客户 <strong>${esc(r.customer_key)}</strong>`;
  $('#go-handle').hidden=handling;
  if(handling){
    setCaseTab('handling');
    const timeline=$('#case-events').closest('[data-case-content]');timeline.hidden=false;
  }else setCaseTab('evidence');
};

route=function(name){
  name=names[name]?name:'overview';
  if(name!==state.route&&state.dirty){dialog('保存当前办理内容','<p>请先保存当前案件材料，再切换工作页面。</p>');return;}
  state.route=name;document.body.dataset.activePage=name;
  $$('[data-page]').forEach(p=>p.hidden=p.dataset.page!==(name==='handling'?'review':name));
  $$('[data-route]').forEach(b=>b.classList.toggle('active',b.dataset.route===name));
  $('#route-label').textContent=names[name];
  if(location.hash!=='#'+name)history.pushState(null,'','#'+name);
  if(name==='overview')renderOverview();
  if(name==='queue')renderQueue();
  if(['review','handling'].includes(name))renderReview();
  if(name==='agents'){renderAgent();renderAgentCenter();}
  renderTasks();window.scrollTo(0,0);
};

const sixPageSelect=select;
select=function(id,open=false){
  sixPageSelect(id,open);
  if(state.route==='overview')renderOverview();
  if(state.route==='agents')renderAgentCenter();
};
const originalClearFilters=$('#clear-filters').onclick;
$('#clear-filters').onclick=()=>{state.riskBand=null;originalClearFilters();};
$('#overview-period').onchange=e=>{state.period=e.target.value;state.drafts={};renderOverview();};
$('#go-handle').onclick=()=>route('handling');
$('#open-copilot').onclick=()=>route('agents');
$('#agent-start').onclick=()=>route('handling');
$('#task-create').onclick=()=>route('handling');
$('#presentation-toggle').onclick=()=>{
  document.body.classList.toggle('showcase');
  $('#presentation-toggle').textContent=document.body.classList.contains('showcase')?'完整视图':'展示视图';
  PAGE_SIZE=document.body.classList.contains('showcase')?5:8;state.page=0;
  if(state.route==='queue')renderQueue();
};
$$('[data-route]').forEach(b=>b.onclick=()=>route(b.dataset.route));
document.addEventListener('click',e=>{
  let b=e.target.closest('[data-go]');if(b)route(b.dataset.go);
  b=e.target.closest('[data-route-scope]');
  if(b){state.view=b.dataset.routeScope;state.riskBand=null;state.workView='all';state.clue=state.search=state.filter='';state.page=0;route('queue');}
  b=e.target.closest('[data-risk-bin]');
  if(b){state.riskBand=+b.dataset.riskBin;state.view='all';state.workView='all';state.clue=state.search=state.filter='';state.page=0;route('queue');}
  b=e.target.closest('[data-overview-clue]');
  if(b){state.riskBand=null;state.view='all';state.workView='all';state.search=state.filter='';state.clue=b.dataset.overviewClue;state.page=0;route('queue');}
});
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&e.target.matches('[data-risk-bin]'))e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));
});
// Applying a generated draft opens the actual case form before editing it.
document.addEventListener('click',e=>{
  if(e.target.closest('#apply-brief')&&state.route==='agents')route('handling');
},true);

Object.assign(window.CreditWorkbench,{route,select,filtered,source,renderOverview,renderAgentCenter,renderQueue,renderReview,riskHistogram});
route(state.route);
