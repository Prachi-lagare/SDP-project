# Warfarin SDP — Final Reliability & Output Edition (V11)

**Research-only application. Do not use for prescribing, adjusting or taking warfarin.**

## Updates from V10

- Retains the original 70/15/15 patient-disjoint split, trained-from-scratch models, ML/DL comparison, SHAP, and model evaluation.
- Numerical output now shows the current historical value, experimental model estimate, and their signed difference.
- A small difference (no larger than the selected model's validation MAE) is explicitly described as **within the validation error scale**, rather than automatically labelled Increase/Decrease. This is a heuristic contextual description, not a calibrated prediction interval.
- The classifier remains a separate retrospective task; no combined prescription is generated.
- Adds clear validation, held-out test, five-fold patient-grouped audit and persistence-baseline comparison beside every prediction.
- Removes misleading automatic seven-day abstract patterns and the duplicated prescription result calendar. Retains one editable Monday–Sunday calendar for displaying an **existing** clinician-approved prescription.
- Source-unit ambiguity is retained honestly. No mg/day claim.

## Setup (PowerShell)

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 and http://127.0.0.1:8000/research-evidence.
To regenerate from the **included anonymized project dataset only**:

```powershell
python -m backend.train
python -m backend.reliability
python -m pytest -q
```

Saved models were created by this project's own training pipeline, not pretrained external models.

## Limitations and actual metrics

- 631 rows; 52 patient IDs; 363 usable visit pairs in source audit.
- Training 36 patients / 288 pairs; validation 8 / 47; test 8 / 28.
- Extra Trees selected as the learned regression model using validation MAE: train 0.2824, validation 0.5402, test 0.3528.
- Logistic Regression selected as historical-change classifier using validation Macro F1: train 47.59%, validation 55.93%, test 41.75%.
- Persistence baseline validation MAE 0.5745, test MAE 0.2143. Patient-grouped five-fold development audit persistence MAE 0.3079 versus Extra Trees MAE 0.4866.
- Therefore a learned regressor has **not demonstrated superiority over the simple benchmark**. The classifier also has weak small-class generalization. The project does not establish clinical usefulness.
- No automatic prescription schedules or vitamin K instructions are generated.

See `PRESENTATION_GUIDE.md`, `artifacts/metrics/`, and `artifacts/research/` for report material.
