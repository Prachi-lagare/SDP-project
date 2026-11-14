"""Optional from-scratch PyTorch MLP retrospective regression experiment.
Does not replace the selected deployment model; validation-driven research only.
Install torch separately. Uses same 36/8/8 patient partition as backend.train.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from backend.train import ROOT,read_pairs,NUM,CAT,FEATS,SEED,scores_reg

def run():
    try:import torch
    except ImportError:raise SystemExit('Optional PyTorch experiment: pip install torch')
    torch.set_num_threads(1);torch.manual_seed(SEED)
    orig,pairs=read_pairs();pids=np.sort(orig.Patient_ID.unique())
    tr,tmp=train_test_split(pids,test_size=16,random_state=SEED)
    va,te=train_test_split(tmp,test_size=8,random_state=SEED)
    ids={'train':tr,'validation':va,'test':te}
    data={k:pairs[pairs.Patient_ID.isin(v)] for k,v in ids.items()}
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],sparse_threshold=0)
    X={k:torch.tensor(prep.fit_transform(d[FEATS]) if k=='train' else prep.transform(d[FEATS]),dtype=torch.float32) for k,d in data.items()}
    y={k:torch.tensor(d.next_dose.to_numpy(dtype=np.float32).reshape(-1,1)) for k,d in data.items()}
    net=torch.nn.Sequential(torch.nn.Linear(X['train'].shape[1],16),torch.nn.ReLU(),torch.nn.Linear(16,8),torch.nn.ReLU(),torch.nn.Linear(8,1))
    opt=torch.optim.Adam(net.parameters(),lr=.005,weight_decay=.15)
    best=float('inf');bad=0;checkpoint=None
    for epoch in range(350):
        net.train();opt.zero_grad();pred=net(X['train']);loss=torch.nn.functional.mse_loss(pred,y['train']);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():val=float(torch.nn.functional.l1_loss(net(X['validation']),y['validation']).item())
        if val < best-1e-4:
            best=val;bad=0;checkpoint={k:v.detach().clone() for k,v in net.state_dict().items()}
        else:bad+=1
        if bad>=40:break
    net.load_state_dict(checkpoint);net.eval()
    with torch.no_grad():report={k:scores_reg(data[k].next_dose.to_numpy(),net(X[k]).numpy().ravel()) for k in data}
    output={'model':'PyTorch MLP, random initialization, 16-8 hidden units','epoch':epoch+1,'metrics':report,'selected_for_website':False,'notes':'Exploratory; test evaluation only after validation stopping. Compare against models selected without viewing test scores.'}
    (ROOT/'artifacts/metrics'/'dl_experiment.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))
if __name__=='__main__':run()
