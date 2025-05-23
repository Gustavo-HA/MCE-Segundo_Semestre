from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

iris = load_iris()
X = np.array(iris.data)
y = np.array(iris.target)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------- Random Forest ---------

rf_param_dist = {
    'n_estimators': [10, 20, 30, 40, 50, 60],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'bootstrap': [True,False]
}

print("--- Random Forest: RandomizedSearchCV ---")
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)
start_time_rf_random = time.time()
rf_random_search.fit(X_train, y_train)
end_time_rf_random = time.time()
print(f"RandomizedSearchCV (Random Forest) took: {end_time_rf_random - start_time_rf_random:.2f} seconds")
best_rf_model_random = rf_random_search.best_estimator_
y_pred_rf_random = best_rf_model_random.predict(X_test)
print("Random Forest (RandomizedSearchCV) - Best Parameters:", rf_random_search.best_params_)
print("\nRandom Forest (RandomizedSearchCV) - Classification Report:")
print(classification_report(y_test, y_pred_rf_random))


print("\n--- Random Forest: GridSearchCV ---")

rf_param_grid = {
    'n_estimators': [10, 20, 30],
    'min_samples_split': [4, 5, 6],
    'min_samples_leaf': [3, 4, 5],
    'max_features': ['log2'],
    'max_depth': [25, 32, 40, None],
    'bootstrap': [False]
}

rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_dist,
    cv=3,
    verbose=0,
    n_jobs=-1
)
start_time_rf_grid = time.time()
rf_grid_search.fit(X_train, y_train)
end_time_rf_grid = time.time()
print(f"GridSearchCV (Random Forest) took: {end_time_rf_grid - start_time_rf_grid:.2f} seconds")
best_rf_model_grid = rf_grid_search.best_estimator_
y_pred_rf_grid = best_rf_model_grid.predict(X_test)
print("Random Forest (GridSearchCV) - Best Parameters:", rf_grid_search.best_params_)
print("\nRandom Forest (GridSearchCV) - Classification Report:")
print(classification_report(y_test, y_pred_rf_grid))


# --------- Logistic Regression ---------
lr_param_dist = {
    'estimator__C': [0.01, 0.1, 1, 10, 50],
    'estimator__penalty': ['l1', 'l2'],
    'estimator__solver': ['liblinear']
}

ovr_lr = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=200))

print("\n\n--- Logistic Regression: RandomizedSearchCV ---")
lr_random_search = RandomizedSearchCV(
    estimator=ovr_lr,
    param_distributions=lr_param_dist,
    n_iter=20,
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)
start_time_lr_random = time.time()
lr_random_search.fit(X_train, y_train)
end_time_lr_random = time.time()
print(f"RandomizedSearchCV (Logistic Regression) took: {end_time_lr_random - start_time_lr_random:.2f} seconds")
best_lr_model_random = lr_random_search.best_estimator_
y_pred_lr_random = best_lr_model_random.predict(X_test)
print("Logistic Regression (RandomizedSearchCV) - Best Parameters:", lr_random_search.best_params_)
print("\nLogistic Regression (RandomizedSearchCV) - Classification Report:")
print(classification_report(y_test, y_pred_lr_random))

print("\n--- Logistic Regression: GridSearchCV ---")


lr_grid_search = GridSearchCV(
    estimator=ovr_lr,
    param_grid=lr_param_dist,
    cv=3,
    verbose=0,
    n_jobs=-1
)
start_time_lr_grid = time.time()
lr_grid_search.fit(X_train, y_train)
end_time_lr_grid = time.time()
print(f"GridSearchCV (Logistic Regression) took: {end_time_lr_grid - start_time_lr_grid:.2f} seconds")
best_lr_model_grid = lr_grid_search.best_estimator_
y_pred_lr_grid = best_lr_model_grid.predict(X_test)
print("Logistic Regression (GridSearchCV) - Best Parameters:", lr_grid_search.best_params_)
print("\nLogistic Regression (GridSearchCV) - Classification Report:")
print(classification_report(y_test, y_pred_lr_grid))