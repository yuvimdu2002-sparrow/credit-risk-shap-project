#credit_shap_project.py
"""
Credit risk modeling + explainability project.
- Generates a synthetic credit dataset (10k rows)
- Trains LogisticRegression, RandomForest, GradientBoosting
- Attempts SHAP analysis (TreeExplainer). If shap import fails, falls back to permutation importance.
- Saves models, CSVs, and plots into ./output/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt

OUTPUT_DIR = "/content/drive/MyDrive/credit_risk_output/" # Changed to a directory path
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

def make_synthetic(n=10000):
    age = np.random.randint(21, 75, size=n)
    annual_income = np.round(np.random.normal(45000, 20000, size=n)).clip(8000, 300000)
    loan_amount = np.round(np.random.normal(15000, 12000, size=n)).clip(500, 200000)
    term_months = np.random.choice([36,48,60,72], size=n, p=[0.5,0.25,0.2,0.05])
    credit_score = np.round(np.random.normal(640,70,size=n)).clip(300,850)
    employment_years = np.round(np.random.exponential(3,size=n)).astype(int)
    debt_to_income = np.round((loan_amount/annual_income)*100,2)
    num_credit_lines = np.random.poisson(4,size=n).clip(0,20)
    past_delinquencies = np.random.poisson(0.3,size=n)
    loan_purpose = np.random.choice(['debt_consolidation','home_improvement','car','small_business','other'],
                                   size=n, p=[0.5,0.2,0.15,0.08,0.07])
    logit = (
        -3.0
        + 0.00002 * loan_amount
        + 0.03 * past_delinquencies
        + 0.02 * debt_to_income
        - 0.006 * credit_score
        - 0.01 * employment_years
        + np.where(term_months>48, 0.2, 0.0)
        + np.where(loan_purpose=='small_business', 0.4, 0.0)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(n) < prob).astype(int)
    df = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'term_months': term_months,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'debt_to_income': debt_to_income,
        'num_credit_lines': num_credit_lines,
        'past_delinquencies': past_delinquencies,
        'loan_purpose': loan_purpose,
        'default': y
    })
    return df

def train_and_save(df):
    X = df.drop(columns=['default'])
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

    numeric_features = ['age','annual_income','loan_amount','term_months','credit_score','employment_years','debt_to_income','num_credit_lines','past_delinquencies']
    categorical_features = ['loan_purpose']
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                      ('cat', categorical_transformer, categorical_features)])

    logreg = Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))])
    rf    = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    gb    = Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))])

    print("Training Logistic Regression...")
    logreg.fit(X_train, y_train)
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    print("Training Gradient Boosting...")
    gb.fit(X_train, y_train)

    # Evaluate
    def eval_model(m, Xt, yt, name):
        preds = m.predict(Xt)
        probs = m.predict_proba(Xt)[:,1]
        acc = accuracy_score(yt, preds)
        auc = roc_auc_score(yt, probs)
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}  ROC AUC: {auc:.4f}")
        print(classification_report(yt, preds, digits=4))
        print("Confusion matrix:\n", confusion_matrix(yt, preds))
        return {'accuracy':acc, 'roc_auc':auc}

    res_log = eval_model(logreg, X_test, y_test, "Logistic Regression")
    res_rf  = eval_model(rf, X_test, y_test, "Random Forest")
    res_gb  = eval_model(gb, X_test, y_test, "Gradient Boosting")

    # Save pipelines
    joblib.dump(logreg, os.path.join(OUTPUT_DIR, "logreg_pipeline.joblib"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_pipeline.joblib"))
    joblib.dump(gb, os.path.join(OUTPUT_DIR, "gb_pipeline.joblib"))

    # Try SHAP; if SHAP not available / errors, fallback to permutation importance
    try:
        import shap
        print("SHAP detected. Running TreeExplainer for Random Forest...")
        pre = rf.named_steps['pre']
        clf = rf.named_steps['clf']

        # Build feature names for SHAP (these correspond to the transformed features)
        ohe = pre.named_transformers_['cat'].named_steps['onehot']
        ohe_names = list(ohe.get_feature_names_out(categorical_features))
        shap_feature_names = numeric_features + ohe_names

        X_test_trans = pre.transform(X_test)
        # Convert X_test_trans to a DataFrame with explicit feature names for SHAP
        X_test_trans_df = pd.DataFrame(X_test_trans, columns=shap_feature_names)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_trans_df)  # Use DataFrame here
        shap_vals_pos = shap_values[1]

        # summary plot
        shap.summary_plot(shap_vals_pos, X_test_trans_df, feature_names=shap_feature_names, show=False) # Use DataFrame here
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_rf.png"), dpi=150)
        plt.close()

        # save top mean(|shap|)
        mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
        fi = pd.DataFrame({'feature': shap_feature_names, 'mean_abs_shap': mean_abs_shap}).sort_values('mean_abs_shap', ascending=False)
        fi.to_csv(os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"), index=False)

        # local explanations: first 5 test rows
        for i in range(5):
            # waterfall plot for sample i
            shap.plots.waterfall(explainer.expected_value[1], shap.Explanation(values=shap_vals_pos[i], base_values=explainer.expected_value[1],
                                                                                data=X_test_trans_df.iloc[i], feature_names=shap_feature_names), show=False) # Use DataFrame here
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"shap_waterfall_{i}.png"), dpi=150)
            plt.close()

        print("SHAP plots saved to", OUTPUT_DIR)
        shap_used = True

    except Exception as e:
        print("SHAP could not be used in this run. Falling back to permutation importance.")
        print("Reason / exception:", repr(e))
               # Permutation importance on RandomForest (using original X_test with preprocessing built into pipeline)
        r = permutation_importance(rf, X_test, y_test, n_repeats=8, random_state=42, n_jobs=1)
        # Feature names for permutation importance should correspond to the ORIGINAL features in X_test
        perm_imp_feature_names = X_test.columns.tolist()
        perm_imp = pd.DataFrame({'feature': perm_imp_feature_names, 'importance_mean': r.importances_mean})
        perm_imp = perm_imp.sort_values('importance_mean', ascending=False)
        perm_imp.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)
        # bar plot
        plt.figure(figsize=(8,6))
        topn = perm_imp.head(12)
        plt.barh(range(len(topn)), topn['importance_mean'][::-1])
        plt.yticks(range(len(topn)), topn['feature'][::-1])
        plt.title("Permutation importance (RandomForest)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "permutation_importance.png"), dpi=150)
        plt.close()
        shap_used = False

    # Save short report
    report = {
        "n_rows": len(df),
        "metrics": {"logreg":res_log, "rf":res_rf, "gb":res_gb},
        "shap_used": shap_used
    }
    pd.Series(report).to_json(os.path.join(OUTPUT_DIR, "summary_report.json"))
    # Save selected customers (first five in test set) for manual review
    selected = X_test.reset_index(drop=True).head(5).copy()
    selected['default_actual'] = y_test.reset_index(drop=True).head(5).values
    selected.to_csv(os.path.join(OUTPUT_DIR, "selected_customers.csv"), index=False)
    print("All done. Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = make_synthetic(10000)
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_credit_data.csv"), index=False) # Updated this line to use OUTPUT_DIR as a directory
    train_and_save(df)
