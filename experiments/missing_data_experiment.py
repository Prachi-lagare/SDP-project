"""Controlled feature masking imputation experiment, training patients only.
No generative model claims; this is a classical imputation benchmark.
"""
import json,numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.model_selection import train_test_split
from backend.train import ROOT,read_pairs,SEED

def run():
    source,pairs=read_pairs();patients=np.sort(source.Patient_ID.unique());tr,tmp=train_test_split(patients,test_size=16,random_state=SEED)
    X=pairs.loc[pairs.Patient_ID.isin(tr),['age','weight','current_inr','current_dose']].to_numpy(dtype=float)
    rng=np.random.default_rng(SEED);known=np.argwhere(np.isfinite(X));selected=known[rng.choice(len(known),size=max(1,int(.15*len(known))),replace=False)]
    masked=X.copy();masked[selected[:,0],selected[:,1]]=np.nan
    results={}
    for name,model in {'Median':SimpleImputer(strategy='median'),'KNN':KNNImputer(n_neighbors=5)}.items():
        # Masking evaluation on development patients, for methodology demo.
        restored=model.fit_transform(masked)
        results[name]=round(float(np.sqrt(np.mean((restored[selected[:,0],selected[:,1]]-X[selected[:,0],selected[:,1]])**2))),5)
    output={'masked_fraction':.15,'feature_columns':['age','weight','current_inr','current_dose'],'reconstruction_rmse':results,'note':'Different feature units mean pooled RMSE is descriptive, not a clinical metric. GenAI not implemented.'}
    (ROOT/'artifacts/metrics'/'missing_data.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':run()
