"""Research robustness audit, development patients ONLY. Does not select or refit deployment models.

Run `python -m backend.reliability`. Does not access held-out patients for fitting.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
from .train import ROOT, SEED, FEATS, LABELS, read_pairs, preprocessor


def regression():
    return {
        'Persistence baseline': None,
        'Extra Trees': ExtraTreesRegressor(n_estimators=130,min_samples_leaf=6,random_state=SEED,n_jobs=-1),
        'Random Forest': RandomForestRegressor(n_estimators=130,min_samples_leaf=6,max_features=.8,random_state=SEED,n_jobs=-1),
        'Ridge': Ridge(alpha=20),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100,learning_rate=.04,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=8,random_state=SEED),
    }


def classification():
    return {
        'Majority baseline':DummyClassifier(strategy='most_frequent'),
        'Logistic Regression':LogisticRegression(max_iter=700,C=.3,class_weight='balanced',random_state=SEED),
    }


def rounded(x):
    return round(float(x),5)


def run():
    allrows,pairs=read_pairs()
    patients=np.sort(allrows['Patient_ID'].unique())
    trainids,tempids=train_test_split(patients,test_size=16,random_state=SEED)
    validids,testids=train_test_split(tempids,test_size=8,random_state=SEED)
    development=set(trainids)|set(validids)
    heldout=set(testids)
    assert development.isdisjoint(heldout)
    df=pairs[pairs.Patient_ID.isin(development)].copy().reset_index(drop=True)
    assert not set(df.Patient_ID).intersection(heldout)
    kfold=GroupKFold(n_splits=5)
    reg_results={name:[] for name in regression()}
    cls_results={name:[] for name in classification()}
    fold_counts=[]
    for fi,(trainidx,validx) in enumerate(kfold.split(df,groups=df.Patient_ID),start=1):
        tr,va=df.iloc[trainidx],df.iloc[validx]
        assert set(tr.Patient_ID).isdisjoint(set(va.Patient_ID))
        fold_counts.append({'fold':fi,'train_patients':int(tr.Patient_ID.nunique()),'validation_patients':int(va.Patient_ID.nunique()),'train_pairs':len(tr),'validation_pairs':len(va)})
        for name,est in regression().items():
            if est is None:
                ptrain=tr.current_dose.to_numpy();pval=va.current_dose.to_numpy()
            else:
                pipe=Pipeline([('prep',preprocessor()),('model',est)])
                pipe.fit(tr[FEATS],tr.next_dose)
                ptrain=pipe.predict(tr[FEATS]);pval=pipe.predict(va[FEATS])
            reg_results[name].append({'train_mae':rounded(mean_absolute_error(tr.next_dose,ptrain)),'validation_mae':rounded(mean_absolute_error(va.next_dose,pval))})
        for name,est in classification().items():
            pipe=Pipeline([('prep',preprocessor()),('model',est)])
            pipe.fit(tr[FEATS],tr.label)
            ptrain=pipe.predict(tr[FEATS]);pval=pipe.predict(va[FEATS])
            cls_results[name].append({'train_macro_f1':rounded(f1_score(tr.label,ptrain,labels=LABELS,average='macro',zero_division=0)),
                                      'validation_macro_f1':rounded(f1_score(va.label,pval,labels=LABELS,average='macro',zero_division=0)),
                                      'validation_accuracy':rounded(accuracy_score(va.label,pval))})
    def summarize(rows,metric,trainmetric):
        val=np.array([r[metric] for r in rows]);tra=np.array([r[trainmetric] for r in rows]);return {'validation_mean':rounded(val.mean()),'validation_std':rounded(val.std(ddof=1)),'train_mean':rounded(tra.mean()),'gap_validation_minus_train':rounded(val.mean()-tra.mean()),'folds':rows}
    out={'description':'Diagnostic 5-fold GroupKFold audit on development patients only; original 8 test patients never fitted or used for model selection.','development_patients':len(development),'heldout_test_patients_excluded':len(heldout),'development_pairs':len(df),'patient_overlap':False,'fold_counts':fold_counts,
         'regression':{n:summarize(v,'validation_mae','train_mae') for n,v in reg_results.items()},
         'classification':{n:summarize(v,'validation_macro_f1','train_macro_f1') for n,v in cls_results.items()}}
    target=ROOT/'artifacts/research';target.mkdir(parents=True,exist_ok=True)
    (target/'reliability.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    pd.DataFrame([{'model':n,**{k:v[k] for k in ('validation_mean','validation_std','train_mean','gap_validation_minus_train')}} for n,v in out['regression'].items()]).sort_values('validation_mean').to_csv(target/'robustness_regression.csv',index=False)
    pd.DataFrame([{'model':n,**{k:v[k] for k in ('validation_mean','validation_std','train_mean','gap_validation_minus_train')}} for n,v in out['classification'].items()]).sort_values('validation_mean',ascending=False).to_csv(target/'robustness_classification.csv',index=False)
    print(json.dumps({'regression':{k:v['validation_mean'] for k,v in out['regression'].items()},'classification':{k:v['validation_mean'] for k,v in out['classification'].items()}},indent=2))
    return out
if __name__=='__main__':run()
