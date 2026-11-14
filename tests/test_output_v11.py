from fastapi.testclient import TestClient
from backend.main import app
client=TestClient(app)
P={"age":38,"weight":67,"gender":"M","diagnosis":"AVR","current_dose":3,"current_inr":2.8}

def test_output_is_research_only():
    r=client.post('/api/predict',json=P)
    assert r.status_code==200,r.text
    v=r.json()
    assert abs(v['estimated_difference']-(v['experimental_estimate']-v['current_recorded_value']))<0.00002
    assert 'regression_validation_mae' in v
    assert 'baseline_group_cv_mae' in v
    assert 'prescription' in v['disclaimer'].lower() or 'prescribing' in v['disclaimer'].lower()
    assert 'dose_schedule' not in v

def test_no_calendar_from_prediction():
    r=client.post('/api/predict',json=P).json()
    assert 'weekly_doses' not in r
    assert r['difference_within_validation_error_scale'] is True

def test_page_has_single_calendar():
    response=client.get('/')
    assert response.status_code==200
    assert response.text.count('id="week"')==1
    assert 'id="simulationWeek"' not in response.text
    assert 'id="resultWeek"' not in response.text

def test_invalid_data_rejected():
    assert client.post('/api/predict',json={**P,'current_inr':-1}).status_code==422
