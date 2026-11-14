# SDP Presentation Guide — Historical Warfarin Dose Modeling

**Project description:** Explainable retrospective analysis of longitudinal warfarin records using machine learning and deep learning, trained from scratch with the provided dataset. Not a prescribing system.

## 1. Opening (about 45 seconds)
"Our objective is to investigate whether patient history, demographics, diagnosis and INR are associated with the next dose recorded in a hospital dataset. We compare conventional models with deep learning, include a strong no-change benchmark, and make model predictions transparent in a web application."

## 2. Dataset and method (about 60 seconds)
- 631 source rows representing 52 unique patient IDs.
- 363 valid consecutive-visit pairs after documented cleaning and chronological matching.
- 36 patients in training; 8 in validation; 8 held out for testing. Exact row percentages vary as visit counts differ.
- Patient_ID used only for grouping. Name removed entirely. Date used for ordering and creating features/targets, not leaked as the next-visit outcome.
- The target is **the next recorded dose**, not an optimal clinical dose; units have not been verified independently.

## 3. Modeling (about 60 seconds)
- Classical regression models include Extra Trees, Random Forest, Ridge, Gradient Boosting, SVR and others; a persistence benchmark predicts the next recorded value will match the current recorded value.
- Classification separately predicts whether the recorded value increased, decreased or stayed the same.
- Deep learning: MLP and GRU were tested from random initialization; neither showed an advantage on validation in this dataset.
- A denoising autoencoder was evaluated for missing-data reconstruction, separately from the dose model.

## 4. Evaluation (about 75 seconds)
**Original held-out evaluation:**
- Extra Trees regressor: train MAE 0.2824; validation MAE 0.5402; test MAE 0.3528.
- Logistic Regression classifier: train Macro F1 47.59%; validation 55.93%; test 41.75%; test accuracy 53.57%.

**NEW development-patient reliability analysis:** 5-fold GroupKFold; the original test patients were excluded from each fold.
- Persistence benchmark mean validation MAE: **0.3079**.
- Random Forest mean validation MAE: **0.4571**.
- Extra Trees mean validation MAE: **0.4866**.
- Logistic Regression mean validation Macro F1: **37.41%**; majority-class baseline **27.78%**.

**Important interpretation:** The persistence baseline outperforms learned regressors on these visits. This indicates substantial outcome continuity and makes changes harder to predict. Extra Trees also has a train–validation gap, suggesting possible overfitting. Do not claim clinical accuracy or a model that is conclusively not overfit.

## 5. Website demonstration (about 90 seconds)
1. Open homepage. Explain each field corresponds to a known input from the dataset; no name field.
2. Enter a retrospective research example with valid age, weight, diagnosis, current recorded dose and INR. These are **illustrations only**, not patients or medication orders.
3. Click "Analyze historical pattern". Describe the result as an experimental next-recorded estimate and separately trained historical change class. Explain that the directions can disagree because models are independent.
4. Click SHAP to demonstrate contributions to the regression estimate (model features, not causal clinical factors).
5. Scroll to model comparison to demonstrate actual train/validation/test metrics and confusion matrix.
6. Click **View validated research evidence** near the top of the homepage, show patient-disjoint folds and the honest baseline comparison.
7. If asked about weekly calendars, explain the approved-prescription display is a separate visualization, not a model-generated treatment schedule.

## 6. Novelty / research contribution (30–45 seconds)
"Our contribution is not a claim that deep learning solves warfarin dosing. It is a reproducible, patient-separated benchmark comparing several model families, with a longitudinal baseline, generalization audit, from-scratch missing-data experimentation, and explainable deployment for research review."

## Panel questions and short answers

**Q: Why does the simple baseline beat the ML models?**
A: Many recorded doses do not change across visits; copying the current value is difficult to beat without more independent patients, richer variables and reliable change-event labels.

**Q: Why use patient-level rather than row-level splitting?**
A: Visits from the same person are correlated. If a patient appears in both training and testing, a model may look more accurate than it is for a new patient.

**Q: Is 78% R² the same as 78% accuracy?**
A: No. R² measures explained variance in continuous values and is not a classification accuracy percentage.

**Q: Is the model overfitting?**
A: The regression model shows a training-to-validation error gap; its generalization is uncertain. We explicitly measured that gap and compared with baselines.

**Q: Why did you try GRU?**
A: Warfarin records are longitudinal, so GRU can process visit history. With limited independent patient sequences, it did not improve validation error; we report this negative result.

**Q: Did GenAI improve the clinical outcome?**
A: No clinical outcome experiment was possible. A denoising autoencoder showed a small reconstruction improvement on a controlled masking task only, not improved patient dosing.

**Q: Can the web app prescribe warfarin?**
A: No. Source dose units, treatment targets, outcomes and external validation are insufficient. It is an educational retrospective research dashboard.

## How to start the project
```
python -m pip install -r requirements.txt
python -m backend.train
python -m backend.reliability
python -m pytest -q
python -m uvicorn backend.main:app --reload
```
Open `http://127.0.0.1:8000`; additional evidence page is `http://127.0.0.1:8000/research-evidence`.
