/* Unified score presentation; all numeric changes derive from loaded case records. */
const legacySource=source;
source=function(id){
 const r=current(),p=r.base_probability,old=r.previous_base_probability;
 if(['corners','quality','S4'].includes(id)){
  dialog('评分更新依据',`<p>参考额度 ${fmt(r.reference_limit,0)} · 周期 ${esc(r.reference_epoch)}。同一 q 用于风险排序，Q = 100 × (1 − q)。</p><table><thead><tr><th>观察期</th><th>基础预测 p</th><th>统一评分 q</th><th>Q 指数</th></tr></thead><tbody><tr><td>提额月</td><td>${fmt(old*100,4)}%</td><td>${fmt(r.previous_risk_probability*100,4)}%</td><td>${fmt(r.previous_quality_index,4)}</td></tr><tr><td>相邻月</td><td>${fmt(p==null?null:p*100,4)}%</td><td>${fmt(r.risk_probability==null?null:r.risk_probability*100,4)}%</td><td>${fmt(r.quality_index,4)}</td></tr></tbody></table><p>${r.score_available?(r.gate_applied?'本期满足七字段近稳条件，允许区间为前次 q ± 1 个百分点。':'本期按最新账务完整更新。'):r.operating_queue+'：先接续账务核验。'}</p>`);return;
 }
 if(!['history','S2','S3','S5','S6'].includes(id)){
  dialog('案件与评分来源',`<p>${esc(r.customer_key)} · ${esc(r.account_key)} · 观察月 M${r.anchor_month}</p><p>统一风险评分：参考尺度244维XGBoost与414维LightGBM融合，独立校准后按七字段近稳条件更新。Q为同一评分的反向百分制显示。</p><p>作业队列：${esc(r.operating_queue)}。原始金额及相对月份沿用Home Credit字段口径。</p>`);return;
 }
 return legacySource(id);
};
const legacyQualityChart=qualityChart;
qualityChart=function(r){if(!r.score_available)return '<div class="empty">'+esc(r.operating_queue)+' · 完成核验后接续评分</div>';return legacyQualityChart(r);};
attributionChart=function(r){
 if(!r.score_available)return '<div class="empty">'+esc(r.operating_queue)+' · 核对账户与观察期账务</div>';
 const a=r.reference_model_attribution,arr=[['基础信息更新',a.information_update_component],['约束修正变化',a.reference_scale_component],['实际 Q 变化',a.total_joint_change]],lim=Math.max(.5,...arr.map(x=>Math.abs(x[1])))*1.15,zero=246;
 let s=line(zero,15,zero,128,'#bfcab7');arr.forEach(([label,v],i)=>{let y=32+i*38,len=Math.abs(v)/lim*90;s+=text(12,y+4,label)+`<rect x="${v>=0?zero:zero-len}" y="${y-7}" width="${Math.max(.6,len)}" height="14" fill="${i===2?'#678673':v<0?'#bd947b':'#9cad98'}"/>`+text(425,y+4,signed(v),'text-anchor="end"');});s+=text(zero,150,'0','text-anchor="middle"');return svg(s,440,165,'基础信息变化与约束修正相加得到实际Q变化');
};
toolRegistry.explain_reference=r=>({previous_q:r.previous_risk_probability,current_q:r.risk_probability,previous_base:r.previous_base_probability,current_base:r.base_probability,display_Q:r.quality_index,reference_limit:r.reference_limit,gate_applied:r.gate_applied,update_decomposition:r.reference_model_attribution,ref:'S4'});
const oldNextAction=nextAction;
nextAction=function(r){if(r.operating_queue==='逾期核对')return '核对逾期天数、应还与入账';if(r.operating_queue==='账务补齐')return '补齐当期账务并接续评分';if(r.operating_queue==='差异核对')return '核对评分更新与账务差异';return oldNextAction(r);};
const unifiedPlan=toolRegistry.draft_review_plan;
toolRegistry.draft_review_plan=r=>{let p=unifiedPlan(r);if(!r.score_available){p.items=[nextAction(r),...p.items].slice(0,4);p.topic=r.operating_queue;}return p;};
const unifiedQueue=renderQueue;
renderQueue=function(){unifiedQueue();$('#queue-context').textContent=`${{risk:'名单复核',shadow:'差异核对',overdue:'逾期核对',missing:'账务补齐',observe:'持续观察',all:'全部客户'}[state.view]} / ${periodName()} / 统一评分与案件接续`;};
const unifiedReview=renderReview;
renderReview=function(){unifiedReview();const r=current();$('#customer-ribbon').insertAdjacentHTML('beforeend',`<span>作业队列 <strong>${esc(r.operating_queue)}</strong></span>`);};
Object.assign(window.CreditWorkbench,{source,renderQueue,renderReview});
route(state.route);
