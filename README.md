# Telecom Customer Churn Prediction

Predicting which telecom customers will leave, and why, with leakage-safe scikit-learn
pipelines, four tuned classifiers and concrete retention recommendations.

**Data:** 5,986 customers, 26.5 % churn (IBM Telco Customer Churn sample).

## Results (held-out test set, 1,198 customers)

| Model | CV AUC (tuned) | Test AUC | Churn recall | Churn precision |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.848 | 0.831 | 0.75 | 0.52 |
| Random Forest | 0.841 | 0.828 | 0.70 | 0.52 |
| **Gradient Boosting** | **0.848** | **0.835** | 0.48 | 0.65 |
| XGBoost | 0.846 | 0.833 | 0.78 | 0.50 |

All four models land within 0.007 test AUC of each other, so the model choice depends on the
business use. **XGBoost and logistic regression** catch about three in four churners and suit
cheap automated outreach. **Gradient Boosting** is the most precise and suits expensive
interventions such as retention offers. CV and test AUC stay close for every model, which
indicates no leakage or overfitting.

## What drives churn

| Driver | Observation | Recommendation |
|---|---|---|
| Contract type | ~42 % churn on month-to-month vs ~3 % on two-year | Discounts to move customers onto longer contracts |
| Tenure | First 12 months carry the highest risk | 90-day onboarding with proactive check-ins |
| Fiber optic | ~41 % churn vs ~19 % on DSL | Investigate service quality; offer SLA guarantees |
| Payment method | Electronic check ~45 % churn | Small discount for switching to auto-pay |
| Security bundle | Online security + tech support customers churn much less | Bundle both add-ons into base plans |

## Approach

1. **Business framing**: why churn is an ML problem, and what a false negative costs
2. **EDA**: target balance, numeric and categorical churn rates, correlations
3. **Feature engineering**: charges per month of tenure, number of add-on services, and a
   security + tech-support bundle flag
4. **Preprocessing inside `Pipeline` / `ColumnTransformer`**: imputation, scaling and one-hot
   encoding are fitted per CV fold, so nothing leaks from validation data
5. **Class imbalance**: class weights (`class_weight="balanced"`, `scale_pos_weight`) rather
   than SMOTE
6. **Model selection**: 5-fold stratified CV baseline, then `GridSearchCV` (ROC-AUC) for
   logistic regression, random forest, gradient boosting and XGBoost
7. **Evaluation** on an untouched 20 % stratified test set: ROC curves, confusion
   matrices, classification reports
8. **Feature importance and business recommendations**, plus a monthly scoring and
   retraining plan

## Run it

```bash
pip install -r requirements.txt
jupyter notebook churn_analysis.ipynb
```

The data is included in `data/`. Random state is fixed for reproducibility.

## Repository layout

```
churn_analysis.ipynb   full workflow with outputs and figures
data/telco_churn.csv   dataset (see data/README.md)
requirements.txt
```

Machine Learning, Project 2. M.Sc. Big Data & AI, SRH University of Applied Sciences, Leipzig.
