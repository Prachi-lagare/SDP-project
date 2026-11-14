"""From-scratch PyTorch GRU sequence experiment; NOT for treatment.
Patient-separated fixed 36/8/8 split. Temporal window up to 3 visits.
Inputs: age, weight, current INR, current recorded dose; target: next recorded dose.
"""
import json,numpy as np
from sklearn.model_selection import train_test_split
from backend.train import ROOT,read_pairs,SEED,scores_reg

def run():
    try:import torch
    except ImportError:raise SystemExit('Install optional PyTorch with pip install torch')
    torch.set_num_threads(1);torch.manual_seed(SEED)
    original,pairs=read_pairs();pids=np.sort(original.Patient_ID.unique())
    tr,tmp=train_test_split(pids,test_size=16,random_state=SEED);va,te=train_test_split(tmp,test_size=8,random_state=SEED)
    ids={'train':set(tr),'validation':set(va),'test':set(te)}
    cols=['age','weight','current_inr','current_dose']
    # Preprocessing parameters learned exclusively from training patients.
    raw=pairs.loc[pairs.Patient_ID.isin(tr),cols].to_numpy(float)
    means=np.nanmedian(raw,axis=0);means=np.where(np.isfinite(means),means,0.)
    filled=np.where(np.isfinite(raw),raw,means)
    std=np.std(filled,axis=0);std=np.maximum(std,.001)
    arrays={k:[] for k in ids};targets={k:[] for k in ids}
    for pid,df in pairs.groupby('Patient_ID'):
        subset=next((key for key,val in ids.items() if pid in val),None)
        if subset is None:continue
        seq=df.sort_values(['dose_date','Visit_ID'])
        # Each pair corresponds to a current visit; only past/current features appear in a window.
        z=seq[cols].to_numpy(float);z=np.where(np.isfinite(z),z,means);z=(z-means)/std
        labels=seq.next_dose.to_numpy(float)
        for i in range(len(z)):
            w=z[max(0,i-2):i+1]
            padded=np.zeros((3,len(cols)),dtype=np.float32);padded[-len(w):]=w.astype(np.float32)
            arrays[subset].append(padded);targets[subset].append(float(labels[i]))
    x={k:torch.tensor(np.stack(v)) for k,v in arrays.items()}
    y={k:torch.tensor(v,dtype=torch.float32).view(-1,1) for k,v in targets.items()}
    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__();self.gru=torch.nn.GRU(input_size=4,hidden_size=10,batch_first=True);self.head=torch.nn.Linear(10,1)
        def forward(self,x):
            z,_=self.gru(x);return self.head(z[:,-1])
    net=GRU();opt=torch.optim.Adam(net.parameters(),lr=.007,weight_decay=.2)
    best=float('inf');bad=0;checkpoint=None
    for epoch in range(250):
        net.train();opt.zero_grad();loss=torch.nn.functional.mse_loss(net(x['train']),y['train']);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():val=float(torch.nn.functional.l1_loss(net(x['validation']),y['validation']).item())
        if val<best-1e-4:
            best=val;bad=0;checkpoint={k:v.detach().clone() for k,v in net.state_dict().items()}
        else:bad+=1
        if bad>=35:break
    net.load_state_dict(checkpoint);net.eval()
    with torch.no_grad():report={k:scores_reg(y[k].numpy().ravel(),net(x[k]).numpy().ravel()) for k in ids}
    result={'model':'GRU from scratch, 3-visit window','epochs':epoch+1,'patient_counts':{k:len(v) for k,v in ids.items()},'sequence_counts':{k:len(v) for k,v in arrays.items()},'metrics':report,'selected_for_website':False,'note':'Exploratory temporal experiment; short independent patient histories limit reliability.'}
    (ROOT/'artifacts/metrics'/'gru_experiment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':run()
