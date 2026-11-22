/* Case workflow: deterministic business rules, shared by the browser and tests. */
(function (root) {
  'use strict';
  const statuses = ['待复核','核对中','待补充','待复审','已核对'];
  const checkNames = {identity:'账户与观察期',ledger:'账务金额与入账',action:'额度动作依据',reference:'参考周期处理'};
  const actorNames = ['复核01','运营01','客服01','复审01'];
  const actionNames = {save:'保存办理',claim:'领取案件',assign:'分派案件',submit:'提交复审',approve:'完成复审',return:'退回补充',reopen:'重新办理',import:'导入记录'};
  function empty(r) {
    return {case_key:r.case_key,customer_key:r.customer_key,account_key:r.account_key,
      source_row_id:r.row_id,reference_epoch:r.reference_epoch,version:0,
      owner:'风险复核岗',assignee:'',status:'待复核',result:'待核对',topic:'额度动作核验',
      due:'',receipt:'',note:'',action_type:'unknown',action_initiator:'unknown',qualification_change:'unknown',action_evidence_id:'',action_evidence_month:'',score_month:Number(r.anchor_month||0),anchor_policy:'pending',
      checks:Object.fromEntries(Object.keys(checkNames).map(k=>[k,{status:'pending',reference:''}])),
      submitted_by:'',reviewed_by:'',review_note:'',created_at:'',updated_at:''};
  }
  function sanitize(row, incoming={}) {
    const rec=empty(row);
    const fields=['owner','assignee','status','result','topic','due','receipt','note','action_type','action_initiator','qualification_change','action_evidence_id','action_evidence_month','anchor_policy','submitted_by','reviewed_by','review_note','created_at','updated_at'];
    for(const k of fields) if(incoming[k]!=null) rec[k]=String(incoming[k]).trim().slice(0,k==='note'?12000:1000);
    rec.version=Number.isInteger(incoming.version)&&incoming.version>=0?incoming.version:0;
    if(!statuses.includes(rec.status)) throw Error('办理状态无效');
    if(rec.assignee&&!actorNames.includes(rec.assignee)) throw Error('请选择工作台中的经办人');
    if(rec.due&&(!/^\d{4}-\d{2}-\d{2}$/.test(rec.due)||!Number.isFinite(Date.parse(rec.due))||new Date(rec.due).toISOString().slice(0,10)!==rec.due)) throw Error('跟进日期格式无效');
    if(!['unknown','policy','reunderwriting','other'].includes(rec.action_type)) throw Error('动作分类无效');
    if(!['pending','retain','refresh','handoff'].includes(rec.anchor_policy)) throw Error('参考周期处理无效');
    for(const k of Object.keys(checkNames)) {
      const c=incoming.checks?.[k];
      if(c) rec.checks[k]={status:['pending','verified','needs'].includes(c.status)?c.status:'pending',reference:String(c.reference||'').trim().slice(0,600)};
    }
    return rec;
  }
  function blockers(rec) {
    const b=[];
    if(!rec.assignee) b.push('指定经办人');
    if(!rec.note) b.push('填写办理说明');
    if(!rec.receipt) b.push('填写凭证或交接回执');
    if(rec.result==='待核对') b.push('选择核对结论');
    const handoff=rec.result==='交由既有流程处理';
    for(const k of (handoff?['identity','ledger']:Object.keys(checkNames))) {
      if(rec.checks[k].status!=='verified'||!rec.checks[k].reference) b.push('核对'+checkNames[k]+'并填写依据');
    }
    if(!handoff&&rec.action_type==='unknown') b.push('确认额度动作分类');
    if(rec.anchor_policy==='pending') b.push('选择参考周期处理');
    if(rec.action_type==='reunderwriting'&&rec.anchor_policy!=='refresh') b.push('重新资质审查应转入参考周期复核');
    if(rec.action_type==='unknown'&&rec.anchor_policy!=='handoff') b.push('动作原因待确认时转交补充依据');
    if(!['unknown','bank','customer','technical'].includes(rec.action_initiator)) b.push('确认动作发起方');
    if(!['unknown','unchanged','updated'].includes(rec.qualification_change)) b.push('确认资质信息状态');
    if(rec.anchor_policy==='retain') {
      if(rec.action_type!=='policy'||rec.action_initiator!=='bank'||rec.qualification_change!=='unchanged') b.push('沿用参考周期须核实银行经营动作及资质信息未更新');
      if(!rec.action_evidence_id) b.push('登记原始动作凭证编号');
      const month=Number(rec.action_evidence_month);
      if(!/^-?\d+$/.test(rec.action_evidence_month)||String(month)!==rec.action_evidence_month||!Number.isInteger(month)||month>rec.score_month) b.push('确认动作依据在评分观察月已可用');
    }
    if(rec.qualification_change==='updated'&&rec.anchor_policy!=='refresh') b.push('资质信息更新须转入新参考周期复核');
    return b;
  }
  function transition(row, before, input, action, actor, now) {
    if(!actorNames.includes(actor)) throw Error('经办标识无效');
    const old=sanitize(row,before||{}),next=sanitize(row,{...old,...input});
    if(!actionNames[action]) throw Error('办理操作无效');
    if(action==='approve'||action==='return') {
      if(old.status!=='待复审') throw Error('请先提交复审');
      if(actor===old.submitted_by) throw Error('请由另一位复核人员完成复审');
      if(!next.review_note) throw Error('请填写复审意见');
      // Reviewing cannot silently replace the submitted evidence or conclusion.
      const memo=next.review_note;Object.assign(next,old);next.review_note=memo;
      if(action==='approve') {const b=blockers(next);if(b.length) throw Error(b.join('；'));next.status='已核对';next.reviewed_by=actor;}
      else {next.status='核对中';next.reviewed_by='';next.submitted_by='';}
    } else if(action==='reopen') {
      if(old.status!=='已核对') throw Error('仅已完成案件可重新办理');
      if(!next.review_note) throw Error('请填写重新办理的原因');
      next.status='核对中';next.submitted_by='';next.reviewed_by='';
    } else {
      if(['待复审','已核对'].includes(old.status)) throw Error('请先退回或重新办理后再修改');
      if(action==='claim') {next.assignee=actor;next.status='核对中';}
      if(action==='assign'&&!next.assignee) throw Error('请选择接续经办人');
      if(action==='submit') {const b=blockers(next);if(b.length) throw Error(b.join('；'));next.status='待复审';next.submitted_by=actor;next.reviewed_by='';next.review_note='';}
      else if(!['待复核','核对中','待补充'].includes(next.status)) throw Error('请通过“提交复审”完成办理');
      if(next.status==='待补充'&&(!next.due||!next.note)) throw Error('待补充案件需要跟进日期和补充事项');
    }
    next.version=old.version+1;next.created_at=old.created_at||now;next.updated_at=now;
    const changed=Object.keys(next).filter(k=>!['version','created_at','updated_at'].includes(k)&&JSON.stringify(next[k])!==JSON.stringify(old[k]));
    return {record:next,event:{case_key:row.case_key,customer_key:row.customer_key,source_row_id:row.row_id,
      action,action_label:actionNames[action],actor,from_status:old.status,status:next.status,owner:next.owner,
      assignee:next.assignee,version:next.version,updated_at:now,changed,note:action==='approve'||action==='return'||action==='reopen'?next.review_note:next.note,topic:next.topic}};
  }
  root.CaseWorkflow={statuses,checkNames,actorNames,actionNames,empty,sanitize,blockers,transition};
  if(typeof module!=='undefined') module.exports=root.CaseWorkflow;
})(typeof window!=='undefined'?window:globalThis);
