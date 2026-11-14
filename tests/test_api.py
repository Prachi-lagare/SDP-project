from fastapi.testclient import TestClient
from backend.main import app
client=TestClient(app)
PAYLOAD={'age':38,'weight':67,'gender':'M','diagnosis':'AVR','current_dose':3,'current_inr':2.8}
def test_home():assert client.get('/').status_code==200
def test_health():assert client.get('/api/health').json()['status']=='ok'
def test_metrics():assert client.get('/api/model-comparison').json()['source_patients']==52
def test_predict():
 r=client.post('/api/predict',json=PAYLOAD);assert r.status_code==200,r.text
 d=r.json();assert 'experimental_estimate' in d and 'models_agree' in d
 assert d['regression_model']==client.get('/api/model-info').json()['selected']['regression']
def test_validation():assert client.post('/api/predict',json={**PAYLOAD,'current_inr':-10}).status_code==422
def test_explain():assert client.post('/api/explain',json=PAYLOAD).status_code==200
