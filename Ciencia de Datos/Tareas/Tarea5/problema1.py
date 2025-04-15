import numpy as np
from typing import Literal
import warnings


class NaiveBayes:
    def __init__(self,
                 kernel: Literal["gaussian", "epanechnikov"] = None):
        self.kernel = kernel
        self.h_calculated_per_class = {}
        if kernel is not None and kernel not in ["gaussian", "epanechnikov"]:
            raise ValueError(
                "Kernel inválido. Elige 'gaussian' o 'epanechnikov'.")

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.classes = classes
        self.p_classes = counts / len(y)

        # Calcular medias y varianzas para cada clase
        if self.kernel is None:
            self.mu = np.array([X[y == c].mean(axis=0) for c in classes])
            self.sigma = np.array([X[y == c].var(axis=0) for c in classes])
        else:
            self._data_classes = {c: X[y == c] for c in classes}
            for c in self.classes:
                n_c = self._data_classes[c].shape[0]
                if n_c > 1:
                    std_dev_c = np.std(self._data_classes[c], axis=0)
                    h_c = 1.06 * std_dev_c * n_c**(-1/5)
                    h_c = np.maximum(h_c, 1e-9)  # Asegurar un h mínimo > 0
                    self.h_calculated_per_class[c] = h_c
                else:
                    self.h_calculated_per_class[c] = None
                    warnings.warn(
                        f"No se pudo calcular h automáticamente para la clase {c} (muestras <= 1).", RuntimeWarning)

        return self

    def _get_proba(self, X, mu, sigma):
        coef = 1 / np.sqrt(2 * np.pi * sigma)
        exp_term = np.exp(-0.5 * ((X - mu) ** 2) / sigma)
        return coef * exp_term

    def _get_proba_kde(self, X, class_):
        if self.kernel == "gaussian":
            return self._get_gaussian_kde(X, class_)
        elif self.kernel == "epanechnikov":
            return self._get_epanechnikov_kde(X, class_)

    def _get_gaussian_kde(self, X_predict, class_):

        # --- Acceder al h calculado para esta clase ---
        h_vector = self.h_calculated_per_class.get(class_)
        if h_vector is None:
            warnings.warn(
                f"Usando h=1.0 (vector) por defecto para clase {class_} (cálculo automático falló).", RuntimeWarning)
            n_features_fallback = X_predict.shape[1]
            h_vector = np.ones(n_features_fallback) * 1.0
        h_vector = np.maximum(h_vector, 1e-9)

        if class_ not in self._data_classes:
            warnings.warn(f"Clase {class_} no encontrada en _data_classes.")
            # Devolver probabilidad baja
            return np.full(X_predict.shape, 1e-100)

        X_train_class = self._data_classes[class_]
        n_features = X_predict.shape[1]
        n_class_samples = X_train_class.shape[0]

        # Comprobar compatibilidad de h_vector 
        if not isinstance(h_vector, np.ndarray) or h_vector.shape != (n_features,):
            raise ValueError(
                f"h calculado para clase {class_} no es un vector compatible.")

        densidades = np.zeros(X_predict.shape)

        for j in range(n_features):
            h_j = h_vector[j]  # Usar el h específico de la característica
            if h_j <= 0:
                h_j = 1e-9  # Evitar división por cero

            X_predict_j = X_predict[:, j]
            X_train_class_j = X_train_class[:, j]
            u = (X_predict_j[:, np.newaxis] -
                 X_train_class_j[np.newaxis, :]) / h_j
            kernel_values = np.exp(-0.5*u**2)/np.sqrt(2*np.pi)
            density_j = np.sum(kernel_values, axis=1) / (n_class_samples * h_j)
            densidades[:, j] = np.maximum(density_j, 1e-100)

        return densidades

    def _get_epanechnikov_kde(self, X_predict, class_):
        h_vector = self.h_calculated_per_class.get(class_)
        if h_vector is None:
            warnings.warn(
                f"Usando h=1.0 (vector) por defecto para clase {class_} (cálculo automático falló).", RuntimeWarning)
            n_features_fallback = X_predict.shape[1]
            h_vector = np.ones(n_features_fallback) * 1.0
        h_vector = np.maximum(h_vector, 1e-9)

        if class_ not in self._data_classes:
            warnings.warn(f"Clase {class_} no encontrada en _data_classes.")
            return np.full(X_predict.shape, 1e-100)

        X_train_class = self._data_classes[class_]
        n_features = X_predict.shape[1]
        n_class_samples = X_train_class.shape[0]

        if n_class_samples == 0:
            return np.full(X_predict.shape, 1e-100)

        if not isinstance(h_vector, np.ndarray) or h_vector.shape != (n_features,):
            raise ValueError(
                f"h calculado para clase {class_} no es un vector compatible.")

        densidades = np.zeros(X_predict.shape)

        for j in range(n_features):
            h_j = h_vector[j]
            if h_j <= 0:
                h_j = 1e-9
            X_predict_j = X_predict[:, j]
            X_train_class_j = X_train_class[:, j]
            u = (X_predict_j[:, np.newaxis] -
                 X_train_class_j[np.newaxis, :]) / h_j
            kernel_values = 0.75 * (1 - u**2) * (np.abs(u) <= 1)
            density_j = np.sum(kernel_values, axis=1) / (n_class_samples * h_j)
            densidades[:, j] = np.maximum(density_j, 1e-100)

        return densidades

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Usaremos logaritmos para mayor estabilidad numérica
        log_posteriors = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            # Logaritmo del prior P(Ck), +e-100 para evitar 0s
            log_prior = np.log(self.p_classes[idx] + 1e-100)

            if self.kernel is None:
                # Chequear si mu/sigma existen (por si clase estaba vacía en fit)
                if idx >= len(self.mu) or np.isnan(self.mu[idx]).any():
                    log_likelihood_c = np.full(n_samples, -np.inf)  # Log(0)
                else:
                    mu_c = self.mu[idx]
                    sigma_c = self.sigma[idx]
                    # Obtener P(xj | Ck)
                    likelihoods_features = self._get_proba(
                        X, mu_c, sigma_c) 
                    # Calcular log P(X | Ck) = Suma log P(xj | Ck)
                    log_likelihood_c = np.sum(
                        # Shape (n_samples,)
                        np.log(likelihoods_features), axis=1)

            else:  # Caso KDE
                likelihoods_features = self._get_proba_kde(
                    X, c) 
                log_likelihood_c = np.sum(
                    np.log(likelihoods_features), axis=1)  
            log_posteriors[:, idx] = log_likelihood_c + log_prior

        # Elegir la clase con el mayor log-posterior
        predictions = self.classes[np.argmax(log_posteriors, axis=1)]
        return predictions


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_iris, load_wine, load_diabetes
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -------- Iris ---------
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    # ------- Wine ---------
    wine = load_wine()
    X_wine = wine.data
    y_wine = wine.target
    
    # ------- Diabetes ---------
    diabetes = load_diabetes()
    X_diabetes = diabetes.data
    y_diabetes = diabetes.target
    
    
    # Inicializar listas para almacenar resultados
    acc_wine = []
    acc_iris = []
    acc_diabetes = []
    mcc_wine = []
    mcc_iris = []
    mcc_diabetes = []
    
    acc_kde_wine = []
    acc_kde_iris = []
    acc_kde_diabetes = []
    mcc_kde_wine = []
    mcc_kde_iris = []
    mcc_kde_diabetes = []
    
    nb_kde = NaiveBayes(kernel="gaussian") # KDE
    nb = NaiveBayes(kernel=None) # Gaussiano
    
    repeticiones = 10
    seed = 42
    np.random.seed(seed) # Reproducibilidad
    for _ in range(repeticiones):
        # Indices de training
        indices_iris = np.random.choice(
            X_iris.shape[0], size=int(X_iris.shape[0]*0.8), replace = False
        )
        
        indices_wine = np.random.choice(
            X_wine.shape[0], size=int(X_wine.shape[0]*0.8), replace = False
        )
        indices_diabetes = np.random.choice(
            X_diabetes.shape[0], size=int(X_diabetes.shape[0]*0.8), replace = False
        )
        
        # Dividir en train y test
        X_train_iris, y_train_iris = X_iris[indices_iris], y_iris[indices_iris]
        X_test_iris, y_test_iris = X_iris[~indices_iris], y_iris[~indices_iris]
        
        X_train_wine, y_train_wine = X_wine[indices_wine], y_wine[indices_wine]
        X_test_wine, y_test_wine = X_wine[~indices_wine], y_wine[~indices_wine]
        
        X_train_diabetes, y_train_diabetes = X_diabetes[indices_diabetes], y_diabetes[indices_diabetes]
        X_test_diabetes, y_test_diabetes = X_diabetes[~indices_diabetes], y_diabetes[~indices_diabetes]
        
        # ------ Iris ------
        nb_kde.fit(X_train_iris, y_train_iris)
        nb.fit(X_train_iris, y_train_iris)
        
        y_pred_iris_kde = nb_kde.predict(X_test_iris)
        y_pred_iris = nb.predict(X_test_iris)
        acc_iris.append(accuracy_score(y_test_iris, y_pred_iris))
        acc_kde_iris.append(accuracy_score(y_test_iris, y_pred_iris_kde))
        mcc_iris.append(matthews_corrcoef(y_test_iris, y_pred_iris))
        mcc_kde_iris.append(matthews_corrcoef(y_test_iris, y_pred_iris_kde))
        
        # ------ Wine ------
        nb_kde.fit(X_train_wine, y_train_wine)
        nb.fit(X_train_wine, y_train_wine)
        
        y_pred_wine_kde = nb_kde.predict(X_test_wine)
        y_pred_wine = nb.predict(X_test_wine)
        acc_wine.append(accuracy_score(y_test_wine, y_pred_wine))
        acc_kde_wine.append(accuracy_score(y_test_wine, y_pred_wine_kde))
        mcc_wine.append(matthews_corrcoef(y_test_wine, y_pred_wine))
        mcc_kde_wine.append(matthews_corrcoef(y_test_wine, y_pred_wine_kde))
        
        # ------ Diabetes ------
        nb_kde.fit(X_train_diabetes, y_train_diabetes)
        nb.fit(X_train_diabetes, y_train_diabetes)
        
        y_pred_diabetes_kde = nb_kde.predict(X_test_diabetes)
        y_pred_diabetes = nb.predict(X_test_diabetes)
        acc_diabetes.append(accuracy_score(y_test_diabetes, y_pred_diabetes))
        acc_kde_diabetes.append(accuracy_score(y_test_diabetes, y_pred_diabetes_kde))
        mcc_diabetes.append(matthews_corrcoef(y_test_diabetes, y_pred_diabetes))
        mcc_kde_diabetes.append(matthews_corrcoef(y_test_diabetes, y_pred_diabetes_kde))
    
    resultados = pd.DataFrame({
        "Dataset": ["Iris"]*20 + ["Wine"]*20 + ["Diabetes"]*20,
        "Modelo": (["Gaussiano"]*10 + ["KDE"]*10)*3,
        "Accuracy": acc_iris + acc_kde_iris + acc_wine + acc_kde_wine + acc_diabetes  + acc_kde_diabetes,
        "MCC": mcc_iris + mcc_kde_iris + mcc_wine + mcc_kde_wine + mcc_diabetes  + mcc_kde_diabetes,
        "Rep": [i for i in range(1, 11)]*6
    })
    
    resultados.to_csv("./p1_resultados.csv", index=False)
    
    
    for dataset in ["Iris", "Wine", "Diabetes"]:
        plt.figure(figsize=(5, 5))
        sns.lineplot(data = resultados[resultados["Dataset"] == dataset], 
                     x = "Rep", y = "Accuracy", hue = "Modelo")
        plt.title(f"Accuracy por repetición - {dataset}")
        plt.savefig(f"./p1_accuracy_{dataset}.pdf", bbox_inches='tight')
    
    for dataset in ["Iris", "Wine", "Diabetes"]:
        plt.figure(figsize=(5, 5))
        sns.lineplot(data = resultados[resultados["Dataset"] == dataset], 
                     x = "Rep", y = "MCC", hue = "Modelo")
        plt.title(f"MCC por repetición - {dataset}")
        plt.savefig(f"./p1_mcc_{dataset}.pdf", bbox_inches='tight')
        
    print("Resultados promedios:\n")
    print("Iris")
    print("Gaussiano:")
    print("Accuracy:", np.mean(acc_iris))
    print("MCC:", np.mean(mcc_iris))
    print("KDE:")
    print("Accuracy:", np.mean(acc_kde_iris))
    print("MCC:", np.mean(mcc_kde_iris))
    print("\nWine")
    print("Gaussiano:")
    print("Accuracy:", np.mean(acc_wine))
    print("MCC:", np.mean(mcc_wine))
    print("KDE:")
    print("Accuracy:", np.mean(acc_kde_wine))
    print("MCC:", np.mean(mcc_kde_wine))
    print("\nDiabetes")
    print("Gaussiano:")
    print("Accuracy:", np.mean(acc_diabetes))
    print("MCC:", np.mean(mcc_diabetes))
    print("KDE:")
    print("Accuracy:", np.mean(acc_kde_diabetes))
    print("MCC:", np.mean(mcc_kde_diabetes))