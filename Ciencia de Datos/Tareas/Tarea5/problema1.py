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
            for c in self.classes:  # Cambiado a self.classes para consistencia
                n_c = self._data_classes[c].shape[0]
                if n_c > 1:
                    std_dev_c = np.std(self._data_classes[c], axis=0)
                    h_c = 1.06 * std_dev_c * n_c**(-1/5)
                    h_c = np.maximum(h_c, 1e-9)  # Asegurar un h mínimo > 0
                    self.h_calculated_per_class[c] = h_c
                else:
                    # Qué hacer si la clase tiene 0 o 1 muestra? No se puede calcular h bien.
                    # Guardar None o un valor por defecto? Guardemos None por ahora.
                    self.h_calculated_per_class[c] = None
                    warnings.warn(
                        f"No se pudo calcular h automáticamente para la clase {c} (muestras <= 1).", RuntimeWarning)

        return self

    def _get_proba(self, X, mu, sigma):
        # Función de densidad de probabilidad para una Gaussiana
        coef = 1 / np.sqrt(2 * np.pi * sigma)
        exp_term = np.exp(-0.5 * ((X - mu) ** 2) / sigma)
        return coef * exp_term

    def _get_proba_kde(self, X, class_, kernel):
        # Elige la función KDE correcta; ya no necesita pasar 'h'
        if kernel == "gaussian":
            # Llama a la función corregida (que accederá a self.h_calculated_per_class)
            return self._get_gaussian_kde(X, class_)
        elif kernel == "epanechnikov":
            # Llama a la función corregida (que accederá a self.h_calculated_per_class)
            return self._get_epanechnikov_kde(X, class_)
        else:
            # Manejo de error por si acaso
            raise ValueError(
                f"Kernel '{kernel}' no soportado en _get_proba_kde")

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

        # --- Comprobar compatibilidad de h_vector ---
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
        # Asegurar que h no sea problemático
        h_vector = np.maximum(h_vector, 1e-9)

        # --- Acceder a _data_classes (ASUMIENDO que existe) ---
        if class_ not in self._data_classes:
            warnings.warn(f"Clase {class_} no encontrada en _data_classes.")
            return np.full(X_predict.shape, 1e-100)

        X_train_class = self._data_classes[class_]
        n_features = X_predict.shape[1]
        n_class_samples = X_train_class.shape[0]

        if n_class_samples == 0:
            return np.full(X_predict.shape, 1e-100)

        # --- Comprobar compatibilidad de h_vector ---
        if not isinstance(h_vector, np.ndarray) or h_vector.shape != (n_features,):
            raise ValueError(
                f"h calculado para clase {class_} no es un vector compatible.")

        densidades = np.zeros(X_predict.shape)

        for j in range(n_features):
            # --- CORRECCIÓN: Usar h_j específico para la característica ---
            h_j = h_vector[j]
            if h_j <= 0:
                h_j = 1e-9

            X_predict_j = X_predict[:, j]
            X_train_class_j = X_train_class[:, j]
            # --- CORRECCIÓN: Usar h_j en el cálculo de u ---
            u = (X_predict_j[:, np.newaxis] -
                 X_train_class_j[np.newaxis, :]) / h_j
            kernel_values = 0.75 * (1 - u**2) * (np.abs(u) <= 1)
            # --- CORRECCIÓN: Usar h_j en la normalización ---
            density_j = np.sum(kernel_values, axis=1) / (n_class_samples * h_j)
            # --- CORRECCIÓN: Evitar ceros en la salida ---
            densidades[:, j] = np.maximum(density_j, 1e-100)

        return densidades

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Usaremos logaritmos para mayor estabilidad numérica
        log_posteriors = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            # Logaritmo del prior P(Ck)
            # (Añadir epsilon por si p_class fuera 0, aunque no debería si la clase existe)
            log_prior = np.log(self.p_classes[idx] + 1e-100)

            if self.kernel is None:
                # Naive Bayes Gaussiano Estándar
                # Chequear si mu/sigma existen (por si clase estaba vacía en fit)
                if idx >= len(self.mu) or np.isnan(self.mu[idx]).any():
                    log_likelihood_c = np.full(n_samples, -np.inf)  # Log(0)
                else:
                    mu_c = self.mu[idx]
                    sigma_c = self.sigma[idx]
                    # Obtener P(xj | Ck)
                    likelihoods_features = self._get_proba(
                        X, mu_c, sigma_c)  # Shape (n_samples, n_features)
                    # Calcular log P(X | Ck) = Suma log P(xj | Ck)
                    log_likelihood_c = np.sum(
                        # Shape (n_samples,)
                        np.log(likelihoods_features), axis=1)

            else:  # Caso KDE
                # Llamar al dispatcher SIN pasar 'h'
                # _get_proba_kde ahora devuelve P(xj | Ck) usando el h interno correcto
                likelihoods_features = self._get_proba_kde(
                    X, c, self.kernel)  # Shape (n_samples, n_features)
                # Calcular log P(X | Ck) = Suma log P(xj | Ck)
                # (likelihoods_features ya no debería ser cero por los np.maximum)
                log_likelihood_c = np.sum(
                    np.log(likelihoods_features), axis=1)  # Shape (n_samples,)

            # Log posterior (no normalizado) ~ log P(X | Ck) + log P(Ck)
            log_posteriors[:, idx] = log_likelihood_c + log_prior

        # Elegir la clase con el mayor log-posterior
        predictions = self.classes[np.argmax(log_posteriors, axis=1)]
        return predictions
