const $=s=>document.querySelector(s);const session=[];let lastInput=null;
async function api(url,opts){const response=await fetch(url,opts);const result=await response.json();if(!response.ok)throw new Error(result.detail?JSON.stringify(result.detail):response.statusText);return result}
function txt(value){return document.createTextNode(String(value))}
function n(value){return Number(value).toFixed(2)}
// Presentation only: does not modify the raw model prediction or prescribe a dose.
function researchEstimate(value){return Number(value).toFixed(2)}
function cell(tr,value){let td=document.createElement('td');td.append(txt(value));tr.appendChild(td)}
$('#patientForm').addEventListener('submit',async ev=>{
  ev.preventDefault();$('#status').textContent='Analyzing historical data…';
  const f=new FormData(ev.target), data={};
  for(const [k,v] of f.entries())data[k]=['gender','diagnosis'].includes(k)?v:(v===''?null:Number(v));
  lastInput=data;
  try{
    const r=await api('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    $('#resultSection').hidden=false;
    $('#currentValue').textContent=n(r.current_recorded_value);
    $('#estimatedValue').textContent=n(r.experimental_estimate);
    $('#classifiedValue').textContent=r.classified_pattern;
    $('#estimateDelta').textContent=(r.estimated_difference>0?'+':'')+n(r.estimated_difference);
    $('#validationMae').textContent=n(r.regression_validation_mae);
    $('#baselineValue').textContent=n(r.baseline_research_estimate);
    $('#disagreement').textContent=r.interpretation+' The independent change classifier says “'+r.classified_pattern+'”; it is evaluated separately and is not a clinical recommendation.';
    const cvModel=r.regression_group_cv_mae===null?'not available':n(r.regression_group_cv_mae);
    const cvBase=r.baseline_group_cv_mae===null?'not available':n(r.baseline_group_cv_mae);
    $('#modelReliability').textContent='Patient-grouped five-fold CV MAE: '+r.regression_model+' = '+cvModel+'; persistence benchmark = '+cvBase+'. '+(r.regression_group_cv_mae!==null&&r.baseline_group_cv_mae!==null&&r.regression_group_cv_mae>r.baseline_group_cv_mae?'The simple benchmark performed better across these folds; do not assume the learned model improves prediction.':'These are development-fold averages, not individual certainty estimates.');
    $('#selectedModels').textContent='Selected numerical model: '+r.regression_model+' (validation MAE '+n(r.regression_validation_mae)+', held-out test MAE '+n(r.regression_test_mae)+'). Historical classifier: '+r.classification_model+' (validation macro F1 '+(100*r.classification_validation_macro_f1).toFixed(1)+'%, test macro F1 '+(100*r.classification_test_macro_f1).toFixed(1)+'%).';
    $('#explainResult').replaceChildren();
    session.push({time:new Date().toLocaleTimeString(),est:researchEstimate(r.experimental_estimate),cls:r.classified_pattern,interpretation:r.difference_within_validation_error_scale?'Within error scale':'Direction uncertain'});
    drawSession();$('#status').textContent='Research analysis complete';
    $('#resultSection').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(e){$('#status').textContent='Error: '+e.message}
});
$('#patientForm').addEventListener('reset',()=>{$('#status').textContent='';lastInput=null});
function drawSession(){const tbody=$('#sessionBody');tbody.replaceChildren();session.forEach((r,i)=>{let tr=document.createElement('tr');[i+1,r.time,r.est,r.cls,r.interpretation].forEach(v=>cell(tr,v));tbody.append(tr)})}
$('#clearSession').onclick=()=>{session.length=0;drawSession()};
$('#explainBtn').onclick=async()=>{if(!lastInput)return;const box=$('#explainResult');box.textContent='Calculating…';try{const r=await api('/api/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lastInput)});box.replaceChildren();const heading=document.createElement('p');heading.textContent=r.method+'. '+r.note;box.append(heading);const scale=Math.max(0.00001,...r.features.map(v=>Math.abs(v.sensitivity)));for(const v of r.features){const row=document.createElement('div');row.className='feature';const name=document.createElement('span');name.textContent=v.feature;const bg=document.createElement('div');bg.className='barbg';const bar=document.createElement('div');bar.className='bar';bar.style.width=(100*Math.abs(v.sensitivity)/scale)+'%';bg.append(bar);const val=document.createElement('span');val.textContent=(v.sensitivity>0?'+':'')+v.sensitivity.toFixed(3);row.append(name,bg,val);box.append(row)}}catch(e){box.textContent='Explanation unavailable: '+e.message}};
const days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];

const week=$('#week');
const weekInputs=[];
function scheduleValues(){return weekInputs.map(i=>i.value.trim()===''?null:Number(i.value));}
function updateWeeklySummary(){
  const vals=scheduleValues();
  const entered=vals.filter(v=>v!==null && Number.isFinite(v));
  $('#daysEntered').textContent=`${entered.length} / 7`;
  $('#weeklyTotalValue').textContent=entered.length?entered.reduce((a,b)=>a+b,0).toFixed(2):'—';
  $('#weeklyAverage').textContent=entered.length?(entered.reduce((a,b)=>a+b,0)/entered.length).toFixed(2):'—';
  $('#weeklyPreview').hidden=true;
  $('#weeklyMessage').textContent='';
}
days.forEach((day)=>{
  const l=document.createElement('label');
  l.textContent=day;
  const input=document.createElement('input');
  input.type='number';input.placeholder='From prescription';input.step='any';input.min='0';input.max='30';
  input.setAttribute('aria-label',day+' existing prescribed amount');
  input.addEventListener('input',updateWeeklySummary);
  l.append(input);week.append(l);weekInputs.push(input);
});
$('#previewSchedule').onclick=()=>{
  const vals=scheduleValues();
  if(vals.some(x=>x===null)||vals.some(x=>!Number.isFinite(x)||x<0)){
    $('#weeklyMessage').textContent='Enter a valid prescribed amount for all seven days to preview.';
    $('#weeklyPreview').hidden=true;return;
  }
  const preview=$('#weeklyPreviewDays');preview.replaceChildren();
  days.forEach((day,i)=>{
    const item=document.createElement('div');item.className='preview-day';
    const d=document.createElement('span');d.textContent=day;
    const value=document.createElement('strong');value.textContent=vals[i].toFixed(2);
    item.append(d,value);preview.append(item);
  });
  $('#weeklyPreview').hidden=false;
  $('#weeklyMessage').textContent='Displayed entered prescription only.';
};
$('#clearSchedule').onclick=()=>{weekInputs.forEach(i=>i.value='');updateWeeklySummary()};
$('#printSchedule').onclick=()=>{
  if($('#weeklyPreview').hidden){$('#weeklyMessage').textContent='Preview a complete prescribed schedule before printing.';return;}
  window.print();
};
(async()=>{try{const m=await api('/api/model-comparison');const o=$('#overview');[['Source patients',m.source_patients],['Usable visit pairs',m.usable_pairs],['Training patients',m.split.train.patients],['Validation patients',m.split.validation.patients],['Test patients',m.split.test.patients]].forEach(([k,v])=>{const d=document.createElement('div');d.className='stat';const strong=document.createElement('strong');strong.textContent=v;d.append(strong,txt(k));o.append(d)});const reg=$('#regTable tbody');Object.entries(m.regression).sort((a,b)=>a[1].validation.mae-b[1].validation.mae).forEach(([name,x])=>{const tr=document.createElement('tr');if(name===m.selected.regression)tr.className='selected';[name,x.train.mae,x.validation.mae,x.test.mae].forEach(v=>cell(tr,typeof v==='number'?v.toFixed(3):v));reg.append(tr)});const cls=$('#clsTable tbody');Object.entries(m.classification).sort((a,b)=>b[1].validation.macro_f1-a[1].validation.macro_f1).forEach(([name,x])=>{const tr=document.createElement('tr');if(name===m.selected.classification)tr.className='selected';[name,x.train.macro_f1,x.validation.macro_f1,x.test.macro_f1,x.test.accuracy].forEach(v=>cell(tr,typeof v==='number'?(100*v).toFixed(1)+'%':v));cls.append(tr)});const a=m.classification[m.selected.classification].confusion_matrix.test;const cm=$('#confusion');cm.className='cm';['Actual / Predicted','Decrease','No change','Increase'].forEach(h=>{const d=document.createElement('div');d.className='head';d.textContent=h;cm.append(d)});a.forEach((row,i)=>{const title=document.createElement('div');title.className='head';title.textContent=['Decrease','No change','Increase'][i];cm.append(title);row.forEach(v=>{const d=document.createElement('div');d.textContent=v;cm.append(d)})});}catch(e){$('#overview').textContent='Model metrics unavailable: '+e.message}})();
