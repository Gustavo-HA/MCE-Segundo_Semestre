import numpy as np
from itertools import product 
from sklearn.naive_bayes import GaussianNB 
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score

diabetes = load_diabetes()
X_data = diabetes.data
y_data = diabetes.target
n_features = X_data.shape[1]

model_class = GaussianNB

def check_shattering(h, model_cls, X_full, n_trials_per_h=10, random_state=None):
    """
    Intenta encontrar un conjunto de h puntos que puedan ser pulverizados.
    """
    n_samples, n_features = X_full.shape
    if h > n_samples:
        print(f"h={h} es mayor que el número de muestras ({n_samples}).")
        return False # No se pueden seleccionar h puntos distintos

    if random_state is not None:
        np.random.seed(random_state) # Para reproducibilidad en la selección de puntos

    print(f"  Intentando encontrar un conjunto pulverizable de {h} puntos ({n_trials_per_h} intentos)...")

    for trial in range(n_trials_per_h):
        # 1. Muestrear h índices distintos
        point_indices = np.random.choice(n_samples, h, replace=False)
        X_subset = X_full[point_indices]

        can_shatter_this_subset = True # Asumir que se puede hasta que falle una etiqueta

        num_labelings = 0
        for labels_tuple in product([0, 1], repeat=h):
            y_subset = np.array(labels_tuple)
            num_labelings += 1

            # Evitar casos degenerados si el modelo los requiere (ej: solo una clase)
            if len(np.unique(y_subset)) < 2 and h > 1:
                try:
                    model_instance = model_cls()
                    model_instance.fit(X_subset, y_subset)
                    preds = model_instance.predict(X_subset)
                    # Incluso si entrena, la predicción puede no ser útil
                    if not np.all(preds == y_subset):
                         can_shatter_this_subset = False
                         break
                except Exception:
                    # Si fit() falla, no puede aprender esta etiqueta
                    can_shatter_this_subset = False
                    break
                continue # Pasar a la siguiente etiqueta si la degenerada funcionó


            # 3. Entrenar el modelo
            try:
                model_instance = model_cls()
                model_instance.fit(X_subset, y_subset)

                # 4. Evaluar en el conjunto de entrenamiento (error cero?)
                y_pred = model_instance.predict(X_subset)

                # Usar accuracy o comparación directa. Accuracy es más robusto a dtypes.
                if accuracy_score(y_subset, y_pred) < 1.0:
                    can_shatter_this_subset = False
                    break # No necesita probar más etiquetas para ESTE conjunto de puntos

            except Exception as e:
                can_shatter_this_subset = False
                break # No puede aprender esta etiqueta

        # 5. Si se pudieron aprender TODAS las 2^h etiquetas para este conjunto...
        if can_shatter_this_subset:
            print(f"  ¡Éxito! Se encontró un conjunto de {h} puntos pulverizable en el intento {trial + 1}.")
            return True # ¡Encontramos un conjunto que se puede pulverizar para este h!

    print(f"  Fallo. No se encontró un conjunto pulverizable de {h} puntos tras {n_trials_per_h} intentos.")
    return False

# BUCLE PRINCIPAL
h = 1
experimental_vc_dim = 0
max_h_to_test = 10 # Límite práctico (2^10 = 1024)
num_trials = 20    # Número de conjuntos de puntos a probar por cada h

while h <= max_h_to_test:
    print(f"\n--- Probando h = {h} ---")
    if check_shattering(h, model_class, X_data, n_trials_per_h=num_trials, random_state=h): # Usar h como seed
        experimental_vc_dim = h
        h += 1
    else:
        print(f"\nNo se pudo pulverizar para h = {h}.")
        break 
else:
    print(f"\nSe pudo pulverizar hasta h = {experimental_vc_dim} (límite probado: {max_h_to_test}).")
    print(f"La dimensión VC experimental es al menos {experimental_vc_dim}.")

print(f"\nDimensión VC Experimental Estimada para '{model_class.__name__}' en Diabetes: {experimental_vc_dim}")