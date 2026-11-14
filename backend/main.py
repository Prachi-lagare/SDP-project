"""Research app. Experimental historical estimates, NOT medical advice."""
from pathlib import Path
import json
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from .train import FEATS,ROOT,LABELS

app=FastAPI(title='Warfarin Retrospective Research',version='1.0.0')
STATIC=ROOT/'frontend'
app.mount('/static',StaticFiles(directory=STATIC),name='static')

class PatientInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    age:float=Field(ge=0,le=120)
    weight:float=Field(gt=0,le=350)
    gender:str=Field(min_length=1,max_length=30)
    diagnosis:str=Field(min_length=1,max_length=80)
    current_inr:float=Field(ge=0.5,le=15)
    current_dose:float=Field(gt=0,le=10)
    prev_inr:float|None=Field(default=None,ge=0.5,le=15)
    prev_dose:float|None=Field(default=None,gt=0,le=10)
    days_from_previous:float|None=Field(default=None,ge=0,le=365)

@lru_cache
def metrics():
    p=ROOT/'artifacts/metrics/results.json'
    if not p.exists():raise HTTPException(503,'Train project models first: python -m backend.train')
    return json.loads(p.read_text())

@lru_cache
def load(task):
    m=metrics();name=m['selected'][task]
    if task=='regression' and name=='Persistence Baseline':return None
    stem=('reg_' if task=='regression' else 'cls_')+name.lower().replace(' ','_')+'.joblib'
    return joblib.load(ROOT/'artifacts/trained_models'/stem)

@app.get('/')
def home():return FileResponse(STATIC/'index.html')

@app.get('/api/health')
def health():return {'status':'ok','dataset':'source-anonymized','use':'retrospective research only'}

@app.get('/api/model-info')
def model_info():
    m=metrics();return {'selected':m['selected'],'split':m['split'],'patients':m['source_patients'],'pairs':m['usable_pairs'],'dose_units':m['dose_units'],'trained_from_scratch':True}

@app.get('/api/model-comparison')
def comparison():return metrics()

@app.get('/api/reliability')
def reliability():
    p=ROOT/'artifacts/research/reliability.json'
    if not p.exists():
        raise HTTPException(503,'Run python -m backend.reliability')
    return json.loads(p.read_text(encoding='utf8'))

@app.get('/research-evidence')
def research_evidence():
    return FileResponse(STATIC/'evidence.html')

@app.get('/api/input-options')
def input_options():
    return {'gender':['M','F','Other','Unknown'],'diagnosis':['AVR','MVR','DVR','MVR/ TTK','Unknown'], 'features':FEATS,'optional':['prev_inr','prev_dose','days_from_previous']}

def input_frame(p:PatientInput):
    data=p.model_dump();return pd.DataFrame([{x:data[x] for x in FEATS}],columns=FEATS)

@app.post('/api/predict')
def predict(p:PatientInput):
    """Two independent retrospective tasks; no clinical dosage or scheduling output."""
    m=metrics(); x=input_frame(p)
    reg=load('regression'); cls=load('classification')
    estimate=float(p.current_dose if reg is None else reg.predict(x)[0])
    label=str(cls.predict(x)[0])
    delta=estimate-p.current_dose
    chosen_reg=m['selected']['regression']; chosen_cls=m['selected']['classification']
    validation_mae=float(m['regression'][chosen_reg]['validation']['mae'])
    baseline=m['regression']['Persistence Baseline']
    # Validation MAE is an aggregate error scale, NOT an individual prediction interval.
    # Do not label negligible numerical differences as a definite historical dose change.
    small_change=abs(delta)<=validation_mae
    if small_change:
        interpretation='Difference is within the model’s average validation error scale; no reliable direction can be inferred.'
        numeric_pattern='Direction uncertain'
        agreement_status='Not assessable'
    else:
        numeric_pattern='Higher historical estimate' if delta>0 else 'Lower historical estimate'
        interpretation='The numerical estimate differs from the entered value, but this is not a validated treatment change.'
        agreement_status='Descriptive only; tasks are not directly comparable'
    reliability_file=ROOT/'artifacts/research/reliability.json'
    reliability=json.loads(reliability_file.read_text()) if reliability_file.exists() else {}
    reg_cv=reliability.get('regression',{})
    selected_cv=reg_cv.get(chosen_reg,{}).get('validation_mean')
    baseline_cv=reg_cv.get('Persistence baseline',{}).get('validation_mean')
    return {
        'experimental_estimate':round(estimate,5),
        'current_recorded_value':round(float(p.current_dose),5),
        'estimated_difference':round(delta,5),
        'numeric_pattern':numeric_pattern,
        'difference_within_validation_error_scale':small_change,
        'interpretation':interpretation,
        'classified_pattern':label,
        'agreement_status':agreement_status,
        'models_agree':False if small_change else (('Increase' in label and delta>0) or ('Decrease' in label and delta<0)),
        'regression_model':chosen_reg,
        'classification_model':chosen_cls,
        'regression_validation_mae':validation_mae,
        'regression_test_mae':m['regression'][chosen_reg]['test']['mae'],
        'baseline_validation_mae':baseline['validation']['mae'],
        'baseline_test_mae':baseline['test']['mae'],
        'baseline_research_estimate':round(float(p.current_dose),5),
        'regression_group_cv_mae':selected_cv,
        'baseline_group_cv_mae':baseline_cv,
        'classification_validation_macro_f1':m['classification'][chosen_cls]['validation']['macro_f1'],
        'classification_test_macro_f1':m['classification'][chosen_cls]['test']['macro_f1'],
        'units':m['dose_units'],
        'disclaimer':'Retrospective research prediction. No prescribing, dosage adjustment, or vitamin K instructions.'
    }

@app.post('/api/explain')
def explain(p:PatientInput):
    model=load('regression')
    if model is None:
        return {'method':'Fixed baseline','features':[],'note':'Persistence has no trained feature effects.'}
    x=input_frame(p)
    # Genuine Tree SHAP for Extra Trees/Random Forest/Gradient Boosting.
    try:
        import shap
        transformed=model.named_steps['prep'].transform(x)
        transformed=transformed.toarray() if hasattr(transformed,'toarray') else np.asarray(transformed)
        tree=model.named_steps['model']
        explainer=shap.TreeExplainer(tree)
        vals=np.asarray(explainer.shap_values(transformed))
        if vals.ndim==3:vals=vals[:,:,0]
        names=model.named_steps['prep'].get_feature_names_out()
        items=[{'feature':str(name).replace('num__','').replace('cat__',''),'sensitivity':round(float(effect),5)} for name,effect in zip(names,vals[0])]
        items.sort(key=lambda e:abs(e['sensitivity']),reverse=True)
        return {'method':'Tree SHAP','features':items[:15], 'note':'These are local contributions to the model estimate, not clinical effects or causal explanations.'}
    except Exception:
        # Clearly identify fallback as NOT SHAP.
        base=float(model.predict(x)[0]);items=[]
        defaults={'age':50.,'weight':65.,'current_inr':2.5,'current_dose':3.,'prev_inr':2.5,'prev_dose':3.,'days_from_previous':30.,'gender':'Unknown','diagnosis':'Unknown'}
        for feature in FEATS:
            altered=x.copy();altered.loc[0,feature]=defaults[feature]
            perturbed=float(model.predict(altered)[0]);items.append({'feature':feature,'sensitivity':round(base-perturbed,4)})
        items.sort(key=lambda v:abs(v['sensitivity']),reverse=True)
        return {'method':'Local input perturbation (SHAP unavailable)','features':items,'note':'Model sensitivities relative to reference inputs, not causal or medical explanations.'}
