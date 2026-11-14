'use strict';
const $=s=>document.querySelector(s);
const mk=(tag,text)=>{const el=document.createElement(tag);el.textContent=text;return el};
const num=x=>Number(x).toFixed(3);
async function go(){
try{
const [res,rob]=await Promise.all([fetch('/api/model-comparison'),fetch('/api/reliability')]);
if(!res.ok||!rob.ok)throw Error('Metrics endpoint unavailable');
const m=await res.json(),r=await rob.json();
$('#records').textContent=m.source_rows;$('#patients').textContent=m.source_patients;$('#pairs').textContent=m.usable_pairs;$('#test').textContent=m.split.test.patients;
function table(id,data,fields,pct){let el=$(id);el.replaceChildren();Object.entries(data).sort((a,b)=>pct?b[1].validation_mean-a[1].validation_mean:a[1].validation_mean-b[1].validation_mean).forEach(([name,obj])=>{let tr=document.createElement('tr');[name,...fields.map(k=>obj[k])].forEach((x,i)=>tr.append(mk('td',typeof x==='number'?(pct?(x*100).toFixed(1)+'%':num(x)):String(x))));el.append(tr)})}
table('#reg-audit',r.regression,['validation_mean','validation_std','train_mean','gap_validation_minus_train'],false);
table('#cls-audit',r.classification,['validation_mean','validation_std','train_mean'],true);
const box=$('#selection');
[['Regression model',m.selected.regression,'Holdout MAE '+num(m.regression[m.selected.regression].test.mae)],['Classification model',m.selected.classification,'Holdout Macro F1 '+(100*m.classification[m.selected.classification].test.macro_f1).toFixed(1)+'%'],['Patient separation','36 / 8 / 8','Train / validation / test'],['Validation reliability','Five patient-disjoint folds','Development patients only']].forEach(([heading,name,detail])=>{const card=document.createElement('article');card.className='evidence-panel';card.append(mk('h3',heading),mk('p',name),mk('small',detail));box.append(card)});
}catch(err){$('#reg-audit').textContent='Could not load audit: '+err.message;$('#cls-audit').textContent='Check backend and rerun reliability experiment.'}
}
go();
