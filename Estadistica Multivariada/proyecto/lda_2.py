from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv('./data/train.csv')

# Separar variable objetivo
y_train = X_train['y']
X_train = X_train.drop(columns=['y'])  # Eliminar 'y' para que no contamine la distancia

lda = LinearDiscriminantAnalysis(priors=[0.5, 0.5])

# Mostraz matriz de confusión
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_test = pd.read_csv('./data/test.csv')
y_test = X_test['y']
X_test = X_test.drop(columns=['y'])  # Eliminar 'y' para que no contamine la distancia


# Entrenar el modelo LDA
lda.fit(X_train, y_train)
print(lda.get_params())
# Predecir en el conjunto de prueba
y_pred = lda.predict(X_test)
# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Imprimir metricas de rendimiento
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Imprimir la matriz de confusión 
print("Matriz de confusión:")
print(cm)


import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

logged_model = 'runs:/93edc32f551340f8b92d4e68ec465982/lda_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
y_pred = loaded_model.predict(pd.DataFrame(X_test))

print(classification_report(y_test, y_pred))
# Imprimir la matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print(loaded_model.get_params())