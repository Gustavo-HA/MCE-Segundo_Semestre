from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv('./data/bank-full.csv', sep=';')

# Separar variable objetivo
y = X['y']
X = X.drop(columns=['y'])  # Eliminar 'y' para que no contamine la distancia
# Paso 2: Convertir variables categóricas a numéricas
X = pd.get_dummies(X, drop_first=True)
# Paso 3: Estandarizar las características
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Paso 4: Aplicar LDA

#lda = LinearDiscriminantAnalysis(n_components=1)  # máximo = clases - 1
lda = LinearDiscriminantAnalysis(priors=[0.5, 0.5])

# Mostraz matriz de confusión
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                    random_state=42, stratify=y)
# Entrenar el modelo LDA
lda.fit(X_train, y_train)
# Predecir en el conjunto de prueba
y_pred = lda.predict(X_test)
# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=['yes', 'no'])

# Imprimir metricas de rendimiento
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Imprimir la matriz de confusión 
print("Matriz de confusión:")
print(cm)