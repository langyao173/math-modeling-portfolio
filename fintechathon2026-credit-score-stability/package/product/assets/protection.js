/* Capacity protection is scored by Python. This layer presents saved receipts. */
const priorScopeProtection=scopeRows;
scopeRows=function(){return state.view==='boundary'?ROWS.filter(r=>r.capacity_guard?.guard_changed):priorScopeProtection();};
$('#scope').insertAdjacentHTML('beforeend','<option value="boundary">名单边界核对</option>');
const protectionNode=document.createElement('section');
protectionNode.className='panel protection-panel';protectionNode.dataset.caseContent='evidence';
$('#quality-panel').closest('.evidence-grid').before(protectionNode);
function guardChart(r){
 const c=r.capacity_guard;if(!c)return '<div class="empty">本案沿用所属队列的评分与核对流程。</div>';
 const vals=[r.previous_risk_probability,r.base_probability,c.temporal_candidate,r.risk_probability],labels=['上期评分','本期基础预测','近稳更新候选','最终统一评分'];
 const max=Math.max(.08,...vals)*1.25,w=560,h=202,left=148,right=472;
 let s='';for(let k=0;k<4;k++){const y=26+k*44,v=vals[k],color=k===3?'#51735d':k===2?'#bc9d7f':'#a1b2a2';s+=text(8,y+4,labels[k],'style="font-size:13px"')+line(left,y,right,y,'#e7ece4')+`<rect x="${left}" y="${y-7}" height="14" width="${v/max*(right-left)}" fill="${color}" rx="2"/>`+text(542,y+4,fmt(v*100,3)+'%','text-anchor="end" style="font-size:13px"');}
 s+=text(left,196,'0','style="font-size:11px"')+text(right,196,fmt(max*100,1)+'%','text-anchor="end" style="font-size:11px"');
 return svg(s,w,h,'上期评分、基础预测、近稳候选和容量保护后最终评分');
}
function limitPanel(r){
 const v=r.limit_views;if(!v)return '';
 const views=[['实际授信使用率',v.actual_limit,v.actual_utilization,v.actual_overlimit?'余额超过实际额度':'按真实授信额度核对',''],['参考尺度读数',v.reference_limit,v.reference_ratio,'用于跨期比较；超过100%不代表实际超限','reference']];
 return '<div class="dual-limit">'+views.map(([title,l,u,desc,cls])=>`<div class="limit-item ${cls}"><b>${title} · ${u==null?'—':fmt(u*100,1)+'%'}</b><div class="limit-track"><i style="width:${u==null?0:Math.min(100,Math.max(0,u*100))}%"></i></div><small>分母 ${amount(l)}</small><p>${desc}</p></div>`).join('')+'</div>';
}
const protectionReview=renderReview;
renderReview=function(){
 protectionReview();const r=current(),c=r.capacity_guard;
 protectionNode.innerHTML=`<div class="panel-head"><div><h2>复核名单与评分更新</h2><p>同一最终评分用于排序与跨期比较</p></div><span class="guard-chip ${c?.guard_changed?'exception':''}">${c?(c.guard_changed?'已执行边界保护':'名单成员已校验'):'所属队列接续'}</span></div><div class="protection-grid"><div class="guard-chart">${guardChart(r)}</div><div class="guard-notes">${c?`<b>${c.batch_size.toLocaleString()} 户 / ${c.quota} 个复核名额</b><br>基础序位 ${c.base_rank} → 最终序位 ${c.final_rank}<br>${r.review_at10?'保留在本期风险复核名单':'按本期基础预测留在观察名单'}<br>${c.guard_changed?'为保持名单成员，本次更新额外放宽 '+fmt(c.relaxation_pp,3)+' 个百分点。':r.gate_applied?'近稳更新遵守常规幅度，名单成员保持一致。':'依据本期账务完整更新。'}<br><button class="plain" data-source="S4">展开批次与评分依据 ↗</button>`:'本案不属于相邻月容量保护批次，按所属队列办理。'}</div></div>${limitPanel(r)}`;
 const ribbon=$('#customer-ribbon');if(c)ribbon.firstElementChild.innerHTML=`本批次序位 <strong>${c.final_rank} / ${c.batch_size}</strong>`;else ribbon.firstElementChild.innerHTML='本批次序位 <strong>—</strong>';
 setCaseTab(state.caseTab);
};
const protectionQueue=renderQueue;
renderQueue=function(){
 protectionQueue();let n=$('#guard-queue-note');if(!n){n=document.createElement('div');n.id='guard-queue-note';n.className='guard-queue-note';$('#queue-evidence').prepend(n);}
 n.innerHTML=`<span>相邻月批次 · ${DATA.protection_summary.batch_size.toLocaleString()} 户 · 268 个名额 · 与基础模型名单逐户一致</span><button id="open-boundary">边界核对 ${DATA.protection_summary.exceptions} 户 ↗</button>`;
 $('#open-boundary').onclick=()=>{state.view='boundary';state.page=0;state.riskBand=null;state.search='';$('#search').value='';renderQueue();};
 if(state.view==='boundary')$('#queue-context').textContent='名单边界核对 / '+periodName()+' / 展开评分更新与动作凭证';
};
const protectionSource=source;
source=function(id){
 const r=current(),c=r.capacity_guard;
 if(['quality','corners','S4'].includes(id)&&c){
  dialog('名单保护与更新依据',`<p>${esc(r.customer_key)} · 批次 ${esc(c.batch_id)} · ${c.batch_size} 户 / ${c.quota} 个名额</p>${guardChart(r)}<p>近稳客户先按前期 ±1 个百分点形成候选读数，再按本批次基础模型名单边界作必要修正。最终 q 同时用于排序与跨期比较；排序使用完整精度。</p><p>本案基础序位 ${c.base_rank}，最终序位 ${c.final_rank}。${c.guard_changed?'必要放宽 '+fmt(c.relaxation_pp,4)+' 个百分点。':'本案无需放宽常规更新区间。'}</p><p>批次成员或容量改变后，需要完成整批重算。</p>${limitPanel(r)}`);return;
 }
 if(id==='S3'){dialog('实际额度与参考尺度',limitPanel(r)+'<p>实际超限依据余额与实际授信额度比较；固定参考额度用于构造跨期可比读数。两种分母分别保留。</p>');return;}
 return protectionSource(id);
};
const protectionExplain=toolRegistry.explain_reference;
toolRegistry.explain_reference=r=>({...protectionExplain(r),capacity_receipt:r.capacity_guard,limit_views:r.limit_views,ref:'S4'});
const protectionPlan=toolRegistry.draft_review_plan;
toolRegistry.draft_review_plan=r=>{
 const out=protectionPlan(r),c=getRec(r);
 if(r.capacity_guard?.guard_changed)out.items=['核对本案名单边界及必要放宽幅度',...out.items].slice(0,4);
 out.action_evidence={initiator:c.action_initiator,qualification_change:c.qualification_change,evidence_id:c.action_evidence_id,available_month:c.action_evidence_month};
 if(c.anchor_policy==='retain'&&WF.blockers(c).length)out.items=['补齐银行动作及评分时可用凭证',...out.items].slice(0,4);
 return out;
};
Object.assign(window.CreditWorkbench,{scopeRows,renderQueue,renderReview,source,guardChart});
route(state.route);
