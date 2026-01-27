# Credit Risk Modeling with SHAP — Short Report

*Dataset:* Synthetic sample (10,000 loans) with features: age, annual_income, loan_amount, term_months, credit_score, employment_years, debt_to_income, num_credit_lines, past_delinquencies, loan_purpose.

*Models trained*
- Logistic Regression (baseline)
- Random Forest (explainability target)
- Gradient Boosting (comparison)

*What the script does*
1. Generates data, trains models, evaluates performance (accuracy, ROC AUC).
2. Attempts SHAP (TreeExplainer) for the Random Forest. If SHAP is not available in the environment, the script automatically computes permutation importance as a fallback.
3. Saves plots and models into ./output/.

*Files produced (after run)*
- output/synthetic_credit_data.csv
- output/logreg_pipeline.joblib
- output/rf_pipeline.joblib
- output/gb_pipeline.joblib
- output/shap_summary_rf.png (if SHAP worked)
- output/shap_waterfall_*.png (if SHAP worked)
- output/permutation_importance.png (fallback if SHAP not available)
- output/selected_customers.csv
- output/summary_report.json

*Insights (expected)*
- High loan_amount, high debt_to_income, low credit_score, and more past_delinquencies increase default probability.
- loan_purpose=small_business and longer term_months also tend to increase risk (as built into the synthetic label function).

*Next steps*
- Replace synthetic data with a real dataset (LendingClub / HomeCredit) for validation.
- Add fairness and calibration analyses.
- Build a small web UI to serve local SHAP explanations to loan officers.
