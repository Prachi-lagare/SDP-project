"""Train from scratch on the supplied anonymized source dataset only.
Research-only retrospective prediction; not appropriate for treatment decisions.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,f1_score,confusion_matrix,precision_recall_fscore_support
warnings.filterwarnings('ignore',category=UserWarning)
ROOT=Path(__file__).resolve().parent.parent
SEED=42
NUM=['age','weight','current_inr','current_dose','prev_inr','prev_dose','days_from_previous']
CAT=['gender','diagnosis']
FEATS=NUM+CAT
LABELS=['Decrease observed','No change observed','Increase observed']

def clean_dose(s):
    if pd.isna(s) or isinstance(s,(pd.Timestamp,np.datetime64)):return np.nan
    try:
        x=float(str(s).strip().replace(' ',''))
        return x if 0 < x <= 10 else np.nan
    except (ValueError,TypeError):return np.nan

def read_pairs():
    df=pd.read_excel(ROOT/'data'/'source_anonymized.xlsx')
    df['dose']=df['Current Warfarin Dose'].map(clean_dose)
    df['inr']=pd.to_numeric(df['Current_INR'],errors='coerce')
    df.loc[~df['inr'].between(0.5,15),'inr']=np.nan
    df['dose_date']=pd.to_datetime(df['Dose_Start_Date'],errors='coerce')
    df['age']=pd.to_numeric(df['Age'],errors='coerce')
    df['weight']=pd.to_numeric(df['Weight (kg)'],errors='coerce')
    df['gender']=df['Gender'].fillna('Unknown').astype(str).str.strip().str.upper()
    df['diagnosis']=df['Diagnosis / Indication'].fillna('Unknown').astype(str).str.strip().str.upper()
    df=df.sort_values(['Patient_ID','dose_date','Visit_ID'],kind='stable').reset_index(drop=True)
    g=df.groupby('Patient_ID',sort=False)
    df['prev_inr']=g['inr'].shift(1)
    df['prev_dose']=g['dose'].shift(1)
    df['days_from_previous']=(df['dose_date']-g['dose_date'].shift(1)).dt.days
    df.loc[~df['days_from_previous'].between(0,365),'days_from_previous']=np.nan
    df['next_dose']=g['dose'].shift(-1)
    df['next_date']=g['dose_date'].shift(-1)
    valid=df['dose'].notna() & df['next_dose'].notna() & df['inr'].notna()
    valid &= df['next_date'].notna() & df['dose_date'].notna() & (df['next_date']>=df['dose_date'])
    pairs=df.loc[valid].copy()
    pairs['current_inr']=pairs['inr'];pairs['current_dose']=pairs['dose']
    delta=pairs['next_dose']-pairs['current_dose']
    pairs['label']=np.select([delta < -1e-8,delta > 1e-8],[LABELS[0],LABELS[2]],default=LABELS[1])
    # Input columns have no future INR, next dose, names or patient IDs.
    return df,pairs

def preprocessor():
    return ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),NUM),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])

def scores_reg(y,p):
    return {'mae':round(float(mean_absolute_error(y,p)),5),'rmse':round(float(np.sqrt(mean_squared_error(y,p))),5),'r2':round(float(r2_score(y,p)),5) if len(y)>1 else None}

def scores_cls(y,p):
    return {'accuracy':round(float(accuracy_score(y,p)),5),'macro_f1':round(float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)),5),'weighted_f1':round(float(f1_score(y,p,labels=LABELS,average='weighted',zero_division=0)),5)}

def train():
    (ROOT/'artifacts/trained_models').mkdir(parents=True,exist_ok=True)
    (ROOT/'artifacts/metrics').mkdir(parents=True,exist_ok=True)
    allrows,pairs=read_pairs()
    patients=np.sort(allrows['Patient_ID'].unique())
    trainids,tempids=train_test_split(patients,test_size=16,random_state=SEED)
    validids,testids=train_test_split(tempids,test_size=8,random_state=SEED)
    splits={'train':trainids,'validation':validids,'test':testids}
    assert len(set(trainids)&set(validids))==0 and len(set(trainids)&set(testids))==0 and len(set(validids)&set(testids))==0
    sets={k:pairs.loc[pairs.Patient_ID.isin(ids)].copy() for k,ids in splits.items()}
    if any(len(v)<2 for v in sets.values()):raise ValueError('Insufficient usable visit pairs in split')
    reg_models={
        'Linear Regression':LinearRegression(), 'Ridge':Ridge(alpha=20),
        'Random Forest':RandomForestRegressor(n_estimators=130,min_samples_leaf=6,max_features=.8,random_state=SEED,n_jobs=-1),
        'Extra Trees':ExtraTreesRegressor(n_estimators=130,min_samples_leaf=6,random_state=SEED,n_jobs=-1),
        'Gradient Boosting':GradientBoostingRegressor(n_estimators=90,learning_rate=.04,max_depth=2,min_samples_leaf=12,random_state=SEED),
        'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=100,learning_rate=.04,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=8,random_state=SEED),
        'SVR':SVR(C=1,epsilon=.2),
        'MLP':MLPRegressor(hidden_layer_sizes=(20,),alpha=15,max_iter=350,early_stopping=True,random_state=SEED),
    }
    cls_models={
        'Majority Baseline':DummyClassifier(strategy='most_frequent'),
        'Logistic Regression':LogisticRegression(max_iter=700,C=.3,class_weight='balanced',random_state=SEED),
        'Random Forest':RandomForestClassifier(n_estimators=130,max_depth=5,min_samples_leaf=6,class_weight='balanced',random_state=SEED,n_jobs=-1),
        'Extra Trees':ExtraTreesClassifier(n_estimators=130,max_depth=5,min_samples_leaf=6,class_weight='balanced',random_state=SEED,n_jobs=-1),
        'Gradient Boosting':GradientBoostingClassifier(n_estimators=75,learning_rate=.04,max_depth=2,min_samples_leaf=12,random_state=SEED),
        'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=90,learning_rate=.04,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=8,random_state=SEED),
        'SVC':SVC(C=.5,class_weight='balanced',random_state=SEED),
        'MLP':MLPClassifier(hidden_layer_sizes=(20,),alpha=15,max_iter=350,early_stopping=True,random_state=SEED),
    }
    metrics={'source_rows':len(allrows),'source_patients':len(patients),'usable_pairs':len(pairs),'source_missing_values':{},'split':{},'regression':{},'classification':{},'dose_units':'Not verified in spreadsheet; values are in source dataset units','label_order':LABELS,'trained_from_scratch':True,'patient_split_seed':SEED}
    for col in ['Age','Gender','Weight (kg)','Diagnosis / Indication','Current Warfarin Dose','Current_INR','Dose_Start_Date','Next_INR_Test_Date']:
        metrics['source_missing_values'][col]=int(pd.read_excel(ROOT/'data'/'source_anonymized.xlsx',usecols=[col])[col].isna().sum())
    for k,group in sets.items():metrics['split'][k]={'patients':len(splits[k]),'pairs':len(group),'label_counts':{z:int((group.label==z).sum()) for z in LABELS}}
    for name,est in reg_models.items():
        pipe=Pipeline([('prep',preprocessor()),('model',est)]).fit(sets['train'][FEATS],sets['train']['next_dose'])
        metrics['regression'][name]={k:scores_reg(v.next_dose,pipe.predict(v[FEATS])) for k,v in sets.items()}
        joblib.dump(pipe,ROOT/'artifacts/trained_models'/('reg_'+name.lower().replace(' ','_')+'.joblib'))
    metrics['regression']['Persistence Baseline']={k:scores_reg(v.next_dose,v.current_dose) for k,v in sets.items()}
    best_reg=min(metrics['regression'],key=lambda k:(metrics['regression'][k]['validation']['mae'],metrics['regression'][k]['train']['mae']))
    for name,est in cls_models.items():
        try:
            pipe=Pipeline([('prep',preprocessor()),('model',est)]).fit(sets['train'][FEATS],sets['train']['label'])
            pred={k:pipe.predict(v[FEATS]) for k,v in sets.items()}
            metrics['classification'][name]={k:scores_cls(v.label,pred[k]) for k,v in sets.items()}
            metrics['classification'][name]['confusion_matrix']={k:confusion_matrix(v.label,pred[k],labels=LABELS).tolist() for k,v in sets.items()}
            joblib.dump(pipe,ROOT/'artifacts/trained_models'/('cls_'+name.lower().replace(' ','_')+'.joblib'))
        except Exception as exc: metrics.setdefault('training_errors',{})[name]=str(exc)
    best_cls=max(metrics['classification'],key=lambda k:(metrics['classification'][k]['validation']['macro_f1'],-abs(metrics['classification'][k]['train']['macro_f1']-metrics['classification'][k]['validation']['macro_f1'])))
    metrics['selected']={'regression':best_reg,'classification':best_cls}
    # Keep raw individual-level splits out of shipped metrics; only anonymous split indices exist in source.
    with (ROOT/'artifacts/metrics'/'results.json').open('w') as f:json.dump(metrics,f,indent=2)
    pd.DataFrame([{'model':name,**{s+'_'+m:v[s][m] for s in ['train','validation','test'] for m in ['mae','rmse','r2']}} for name,v in metrics['regression'].items()]).to_csv(ROOT/'artifacts/metrics'/'regression.csv',index=False)
    pd.DataFrame([{'model':name,**{s+'_'+m:v[s][m] for s in ['train','validation','test'] for m in ['accuracy','macro_f1','weighted_f1']}} for name,v in metrics['classification'].items()]).to_csv(ROOT/'artifacts/metrics'/'classification.csv',index=False)
    print(json.dumps({'split':metrics['split'],'selected':metrics['selected'],'selected_regression_metrics':metrics['regression'][best_reg],'selected_classification_metrics':metrics['classification'][best_cls]},indent=2))
    return metrics
if __name__=='__main__':train()
