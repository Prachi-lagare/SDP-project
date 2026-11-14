"""From-scratch denoising autoencoder for numeric feature reconstruction.
Only training patients used for fit; validation patient values used for score.
Never treat reconstructed fields as clinically verified measurements.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer,KNNImputer
from backend.train import ROOT,read_pairs,SEED

def run():
    try:import torch
    except ImportError:raise SystemExit('Optional experiment: pip install torch')
    torch.set_num_threads(1);torch.manual_seed(SEED)
    original,pairs=read_pairs();ids=np.sort(original.Patient_ID.unique());tr,tmp=train_test_split(ids,test_size=16,random_state=SEED);va,te=train_test_split(tmp,test_size=8,random_state=SEED)
    features=['age','weight','current_inr','current_dose']
    rawtrain=pairs[pairs.Patient_ID.isin(tr)][features].to_numpy(float)
    rawval=pairs[pairs.Patient_ID.isin(va)][features].to_numpy(float)
    median=np.nanmedian(rawtrain,axis=0);mean=np.where(np.isfinite(rawtrain),rawtrain,median).mean(axis=0)
    stdev=np.maximum(np.where(np.isfinite(rawtrain),rawtrain,median).std(axis=0),.05)
    xtr=np.where(np.isfinite(rawtrain),rawtrain,median);xval=np.where(np.isfinite(rawval),rawval,median)
    ztr=(xtr-mean)/stdev;zval=(xval-mean)/stdev
    rng=np.random.default_rng(SEED)
    eligible=np.argwhere(np.isfinite(rawval));masked=eligible[rng.choice(len(eligible),size=max(1,int(.15*len(eligible))),replace=False)]
    eval_inputs=zval.copy();eval_inputs[masked[:,0],masked[:,1]]=0.
    class AE(torch.nn.Module):
        def __init__(self):
            super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(4,12),torch.nn.ReLU(),torch.nn.Linear(12,3),torch.nn.ReLU(),torch.nn.Linear(3,12),torch.nn.ReLU(),torch.nn.Linear(12,4))
        def forward(self,x):return self.net(x)
    model=AE();opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=.03)
    target=torch.tensor(ztr,dtype=torch.float32)
    for epoch in range(180):
        corruption=rng.uniform(size=ztr.shape)<.2
        noised=ztr.copy();noised[corruption]=0.
        pred=model(torch.tensor(noised,dtype=torch.float32));loss=torch.nn.functional.mse_loss(pred,target)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():pred=model(torch.tensor(eval_inputs,dtype=torch.float32)).numpy()
    err=lambda z:float(np.sqrt(np.mean((z[masked[:,0],masked[:,1]]-zval[masked[:,0],masked[:,1]])**2)))
    # Baselines train on train rows only and evaluate same validation masking.
    vals=xval.copy();vals[masked[:,0],masked[:,1]]=np.nan
    median_fit=SimpleImputer(strategy='median').fit(rawtrain)
    knn_fit=KNNImputer(n_neighbors=5).fit(rawtrain)
    median_result=(median_fit.transform(vals)-mean)/stdev
    knn_result=(knn_fit.transform(vals)-mean)/stdev
    report={'metric':'normalized masked-entry reconstruction RMSE (lower better)','validation_mask_fraction':.15,'features':features,'scores':{'Train-only median':round(err(median_result),4),'Train-only KNN':round(err(knn_result),4),'From-scratch denoising autoencoder':round(err(pred),4)},'note':'Only 4 numeric columns. This does not evaluate clinical missing INR safety, downstream dose outcomes or external generalization.'}
    (ROOT/'artifacts/metrics'/'genai_imputation_experiment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
