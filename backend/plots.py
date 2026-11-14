"""Produces plots from saved model evaluation, without retraining or altering scores."""
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .train import ROOT

def main():
    d=json.loads((ROOT/'artifacts/metrics/results.json').read_text());out=ROOT/'artifacts/plots';out.mkdir(exist_ok=True,parents=True)
    reg=sorted(d['regression'].items(),key=lambda p:p[1]['validation']['mae']);names=[n for n,_ in reg];v=[s['validation']['mae'] for _,s in reg];t=[s['train']['mae'] for _,s in reg]
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(names));ax.bar(x-.2,t,.4,label='Train');ax.bar(x+.2,v,.4,label='Validation');ax.set_xticks(x,names,rotation=40,ha='right');ax.set_ylabel('MAE, source units');ax.legend();ax.set_title('Regression train vs validation: generalization gap');fig.tight_layout();fig.savefig(out/'regression_generalization.png',dpi=150);plt.close(fig)
    cls=sorted(d['classification'].items(),key=lambda p:-p[1]['validation']['macro_f1']);names=[n for n,_ in cls];v=[s['validation']['macro_f1'] for _,s in cls];t=[s['train']['macro_f1'] for _,s in cls]
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(names));ax.bar(x-.2,t,.4,label='Train');ax.bar(x+.2,v,.4,label='Validation');ax.set_xticks(x,names,rotation=40,ha='right');ax.set_ylabel('Macro F1');ax.legend();ax.set_title('Historical change classification');fig.tight_layout();fig.savefig(out/'classification_generalization.png',dpi=150);plt.close(fig)
    name=d['selected']['classification'];cm=np.array(d['classification'][name]['confusion_matrix']['test']);fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(3),['Decrease','No change','Increase']);ax.set_yticks(range(3),['Decrease','No change','Increase']);ax.set_xlabel('Predicted historical class');ax.set_ylabel('Recorded historical class');ax.set_title('Test confusion matrix — '+name)
    for i in range(3):
      for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
