from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
import pandas as pd
from sklearn.metrics import (classification_report,
                             f1_score,
                             accuracy_score,
                             precision_score,
                             recall_score)
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow
import mlflow.sklearn
import warnings

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bank-market-experiment")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.discriminant_analysis")
warnings.filterwarnings("ignore", category=RuntimeWarning)

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

X_train, y_train = train_df.drop(columns=['y']), train_df['y']
X_test, y_test = test_df.drop(columns=['y']), test_df['y']



# For LDA
space_lda = {
    'solver': hp.choice('solver_lda', ["svd"]),
    'n_components': hp.choice('n_components_lda', [None, 1]),
    "priors": hp.choice('priors_lda', [[0.5, 0.5]]) # Default priors
}

def objective_lda(params):
    # `params` will already have the resolved choices from `hp.choice`
    # e.g., params['shrinkage'] will be None, 'auto', or a float value.
    with mlflow.start_run(nested=True) as run: # nested=True is good practice
        mlflow_run_id = run.info.run_id
        mlflow.set_tag("model_type", "LDA")
        mlflow.log_params(params) # Log the actual parameters hyperopt is trying

        solver = params['solver']
        
        try:
            model = LinearDiscriminantAnalysis(
                solver=solver,
                n_components=params['n_components'],
                priors = params['priors']
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics (assuming 1 is the positive class)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

            mlflow.log_metric("f1_score_class1", f1)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_class1", prec)
            mlflow.log_metric("recall_class1", rec)
            
            report_str = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
            mlflow.log_text(report_str, "classification_report.txt")
            mlflow.sklearn.log_model(model, "lda_model")
            
            # Hyperopt minimizes, so we return the negative of F1 score (our maximization target)
            loss = -f1 
            status = STATUS_OK
        except ValueError as ve: 
            # Catch specific errors, e.g., invalid parameter combinations not caught by logic above
            print(f"LDA ValueError: {ve} with params {params}")
            loss = 1.0 # Max loss (F1=0 means -F1=0, but we want to penalize errors more)
                       # Assigning a high loss value if f1 is between 0 and 1.
            status = STATUS_OK # Let hyperopt continue searching
            f1 = 0.0 # For the results dictionary
        except Exception as e:
            print(f"LDA Generic Exception: {e} with params {params}")
            loss = 1.0 
            status = STATUS_OK
            f1 = 0.0

        return {'loss': loss, 'status': status, 'f1_score': f1, 'mlflow_run_id': mlflow_run_id}

# --- 4. QDA Hyperparameter Optimization ---
space_qda = {
    'reg_param': hp.uniform('reg_param_qda', 0.0, 1.0),
    'tol': hp.loguniform('tol_qda', np.log(1e-5), np.log(1e-2)),
    # `priors` can also be tuned if needed, default is None (estimated from data)
    "priors": hp.choice('priors_qda', [None, [0.5, 0.5]])
}

def objective_qda(params):
    with mlflow.start_run(nested=True) as run:
        mlflow_run_id = run.info.run_id
        mlflow.set_tag("model_type", "QDA")
        mlflow.log_params(params)
        
        try:
            model = QuadraticDiscriminantAnalysis(
                reg_param=params['reg_param'],
                tol=params['tol']
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

            mlflow.log_metric("f1_score_class1", f1)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_class1", prec)
            mlflow.log_metric("recall_class1", rec)

            report_str = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
            mlflow.log_text(report_str, "classification_report.txt")
            mlflow.sklearn.log_model(model, "qda_model")
            
            loss = -f1
            status = STATUS_OK
        except Exception as e:
            print(f"QDA Exception: {e} with params {params}")
            loss = 1.0 
            status = STATUS_OK
            f1 = 0.0
            
        return {'loss': loss, 'status': status, 'f1_score': f1, 'mlflow_run_id': mlflow_run_id}

# --- 5. Run Optimization ---
MAX_EVALS = 10 # Number of iterations for hyperparameter search (increase for more thorough search)

# LDA Optimization
print(f"\n--- Starting LDA Hyperparameter Optimization ({MAX_EVALS} evals) ---")
lda_trials = Trials()
best_lda_raw_hyperparams = fmin(
    fn=objective_lda,
    space=space_lda,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=lda_trials,
    rstate=np.random.default_rng(42) # For reproducible search
)
# `space_eval` converts the indices from `fmin` (for hp.choice) to actual values
best_lda_evaluated_params = space_eval(space_lda, best_lda_raw_hyperparams)
print("\nBest LDA hyperparams (raw from fmin):", best_lda_raw_hyperparams)
print("Best LDA hyperparams (evaluated by space_eval):", best_lda_evaluated_params)

# Find the best trial object to get its F1 score and MLflow run ID
if len(lda_trials.results) > 0 and 'loss' in lda_trials.results[0]:
    best_lda_trial = sorted(lda_trials.results, key=lambda x: x['loss'])[0]
    print(f"Best LDA trial F1-score: {best_lda_trial['f1_score']:.4f}, MLflow Run ID: {best_lda_trial['mlflow_run_id']}")
else:
    print("LDA optimization did not yield any successful trials.")


# QDA Optimization
print(f"\n--- Starting QDA Hyperparameter Optimization ({MAX_EVALS} evals) ---")
qda_trials = Trials()
best_qda_raw_hyperparams = fmin(
    fn=objective_qda,
    space=space_qda,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=qda_trials,
    rstate=np.random.default_rng(42) # For reproducible search
)
best_qda_evaluated_params = space_eval(space_qda, best_qda_raw_hyperparams)
print("\nBest QDA hyperparams (raw from fmin):", best_qda_raw_hyperparams)
print("Best QDA hyperparams (evaluated by space_eval):", best_qda_evaluated_params)

if len(qda_trials.results) > 0 and 'loss' in qda_trials.results[0]:
    best_qda_trial = sorted(qda_trials.results, key=lambda x: x['loss'])[0]
    print(f"Best QDA trial F1-score: {best_qda_trial['f1_score']:.4f}, MLflow Run ID: {best_qda_trial['mlflow_run_id']}")
else:
    print("QDA optimization did not yield any successful trials.")

print("\nHyperparameter optimization complete. Check MLflow UI for detailed logs.")
print(f"MLflow Experiment Name: bank-market-experiment")