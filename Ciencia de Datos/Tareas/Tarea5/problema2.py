import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
import warnings


class DiscretizadorBandWidth(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        if n_bins < 1:
            raise ValueError("n_bins debe ser al menos 1")
        
        self.bordes_intervalo_ = None

    def fit(self, X, y=None):
        """
        Calcula bordes de los intervalos para discretizar los datos continuos.

        Args:
            x (np.ndarray o pd.Series): Datos continuos (una sola característica).
                                         Debe ser convertible a un array 1D de numpy.
            y (None): Ignorado. Por scikit-learn.

        Returns:
            self: Objeto ajustado.
        """
        x = check_array(X, ensure_2d=False, dtype="numeric", ensure_all_finite=True)
        if x.ndim != 1:
             raise ValueError("La entrada 'x' debe ser unidimensional.")

        valor_min = np.min(x)
        valor_max = np.max(x)
        
        self.bordes_intervalo_ = np.linspace(valor_min, valor_max, self.n_bins + 1)
        return self


    def transform(self, X):
        """
        Aplica la discretización usando los bordes aprendidos en fit.

        Args:
            x (np.ndarray o pd.Series): Datos continuos a transformar.

        Returns:
            np.ndarray: Array numpy con los índices de los intervalos (0 a k-1).
        """
        check_is_fitted(self, "bordes_intervalo_")
        x = check_array(X, ensure_2d=False, dtype="numeric", ensure_all_finite=True)
        if x.ndim != 1:
             raise ValueError("La entrada 'x' debe ser unidimensional.")
        
        x_discretizado = pd.cut(x, bins=self.bordes_intervalo_,
                                include_lowest=True, labels=False,
                                duplicates="drop")
        
        x_discretizado = x_discretizado.astype(float)
        if not np.isnan(x_discretizado).any():
            x_discretizado = x_discretizado.astype(int)
        
        return x_discretizado

class DiscretizadorChiMerge(BaseEstimator, TransformerMixin):
    """
    Discretiza características continuas usando el algoritmo ChiMerge (χ²).

    Args:
        max_intervalos (int): Número máximo de intervalos deseados después de la fusión.
    """
    def __init__(self, max_intervalos: int = 5):
        if max_intervalos < 1:
            raise ValueError("max_intervalos debe ser al menos 1.")
        self.max_intervalos = max_intervalos
        self.bordes_intervalo_ = None
        self.num_intervalos_ = None

    def _calcular_chi2(self, tabla_contingencia: np.ndarray) -> float:
        """Calcula el valor chi-cuadrado para una tabla de contingencia."""
        try:
             # Asegurar dimensiones mínimas efectivas
            if tabla_contingencia.shape[0] < 2 or tabla_contingencia.shape[1] < 2:
                 return np.inf # No fusionar

            # Filtrar filas/columnas con suma cero
            filas_validas = np.sum(tabla_contingencia, axis=1) > 0
            columnas_validas = np.sum(tabla_contingencia, axis=0) > 0
            if np.sum(filas_validas) < 2 or np.sum(columnas_validas) < 2:
                 return np.inf # No quedan suficientes válidas

            sub_tabla = tabla_contingencia[filas_validas][:, columnas_validas]

            if sub_tabla.shape[0] < 2 or sub_tabla.shape[1] < 2:
                 return np.inf # Aún inválida después de filtrar

            chi2_val, _, _, _ = chi2_contingency(sub_tabla, correction=False)
            return chi2_val
        except ValueError:
            return np.inf # Si chi2_contingency falla

    def fit(self, x, y):
        """
        Aprende los bordes de los intervalos usando el algoritmo ChiMerge.

        Args:
            x (np.array o pd.Series): Datos continuos (una característica).
            y (np.array o pd.Series): Etiquetas de clase correspondientes.

        Returns:
            self: El objeto discretizador ajustado.
        """
        # Validar y convertir entradas
        x, y = check_X_y(np.array(x).reshape(-1, 1), y, ensure_2d=False, dtype="numeric", ensure_all_finite=True)
        x = x.ravel()

        df = pd.DataFrame({'x': x, 'y': y}).sort_values(by='x').reset_index(drop=True)
        clases_unicas = np.unique(y)

        # Inicializar intervalos
        intervalos = []
        for valor, grupo in df.groupby('x'):
             conteos = grupo['y'].value_counts().reindex(clases_unicas, fill_value=0)
             intervalos.append({
                 'valor_min': valor,
                 'valor_max': valor,
                 'conteos': conteos.values # Array numpy
             })

        # Fusión iterativa
        if self.max_intervalos >= len(intervalos):
            warnings.warn(f"max_intervalos ({self.max_intervalos}) >= número de intervalos iniciales ({len(intervalos)}). No se realizará ninguna fusión significativa.")
        else:
             while len(intervalos) > self.max_intervalos:
                 min_chi2 = np.inf
                 indice_fusion = -1

                 for i in range(len(intervalos) - 1):
                     tabla_contingencia = np.vstack([intervalos[i]['conteos'], intervalos[i+1]['conteos']])
                     chi2 = self._calcular_chi2(tabla_contingencia)
                     if chi2 < min_chi2:
                         min_chi2 = chi2
                         indice_fusion = i

                 if indice_fusion == -1 or min_chi2 == np.inf:
                     break # No se puede fusionar más

                 # Realizar la fusión
                 intervalo_fusionado = {
                     'valor_min': intervalos[indice_fusion]['valor_min'],
                     'valor_max': intervalos[indice_fusion + 1]['valor_max'],
                     'conteos': intervalos[indice_fusion]['conteos'] + intervalos[indice_fusion + 1]['conteos']
                 }
                 intervalos = intervalos[:indice_fusion] + [intervalo_fusionado] + intervalos[indice_fusion + 2:]

        if not intervalos:
             self.bordes_intervalo_ = np.array([-np.inf, np.inf])
        elif len(intervalos) == 1:
             self.bordes_intervalo_ = np.array([-np.inf, np.inf])
        else:
             bordes_intermedios = np.array([intervalo['valor_max'] for intervalo in intervalos[:-1]])
             min_total = df['x'].min()
             max_total = df['x'].max()
             bordes = np.unique(np.concatenate(([min_total], bordes_intermedios, [max_total])))
             bordes[0] = -np.inf
             bordes[-1] = np.inf
             self.bordes_intervalo_ = bordes

        self.num_intervalos_ = len(self.bordes_intervalo_) - 1
        self._is_fitted = True
        return self

    def transform(self, x) -> np.ndarray:
        """Aplica la discretización ChiMerge usando los bordes aprendidos."""
        check_is_fitted(self, 'bordes_intervalo_')
        x_input = check_array(x, ensure_2d=False, dtype="numeric", ensure_all_finite=False)
        if x_input.ndim != 1:
            raise ValueError("La entrada 'x' debe ser unidimensional.")

        x_series = pd.Series(x_input)
        x_discretizado = pd.cut(x_series, bins=self.bordes_intervalo_, labels=False,
                                include_lowest=True, duplicates='drop')

        # Manejo de tipos y NaNs
        x_discretizado = x_discretizado.to_numpy().astype(int)

        return x_discretizado

class DiscretizadorCAIM(BaseEstimator, TransformerMixin):
    """
    Discretiza características continuas usando el algoritmo CAIM.
    Intenta determinar el número óptimo de intervalos maximizando la
    interdependencia clase-atributo (CAIM).
    """
    def __init__(self):
        self.bordes_intervalo_ = None
        self.num_intervalos_ = None

    def _calcular_valor_caim(self, limites: list, x: np.ndarray, y: np.ndarray, clases_unicas: np.ndarray) -> float:
        """Calcula el valor CAIM para un conjunto dado de límites."""
        limites_completos = np.unique(np.concatenate(([-np.inf], limites, [np.inf])))
        num_intervalos = len(limites_completos) - 1
        if num_intervalos == 0: return 0.0

        suma_caim = 0.0
        num_muestras_total = len(x)

        for i in range(num_intervalos):
            inferior = limites_completos[i]
            superior = limites_completos[i+1]
            # Máscara booleana
            mascara = (x > inferior) & (x <= superior)
            if i == 0: # Primer intervalo incluye todo hasta el primer límite
                 mascara = (x <= superior)

            y_intervalo = y[mascara]
            Mr = len(y_intervalo) # M_r: número de puntos en el intervalo r

            if Mr == 0: continue

            # Encontrar clase mayoritaria (max_r)
            clases_intervalo, conteos = np.unique(y_intervalo, return_counts=True)
            if len(conteos) == 0:
                 max_r = 0
            else:
                max_r = np.max(conteos)

            suma_caim += (max_r * max_r) / Mr # Suma de (max_r^2 / M_r)

        return suma_caim / num_intervalos if num_intervalos > 0 else 0.0

    def fit(self, x, y):
        """
        Aprende los bordes de los intervalos usando el algoritmo CAIM.

        Args:
            x (array-like or pd.Series): Datos continuos (una característica).
            y (array-like or pd.Series): Etiquetas de clase correspondientes.

        Returns:
            self: El objeto discretizador ajustado.
        """
        # Validar y convertir entradas
        x, y = check_X_y(np.array(x).reshape(-1, 1), y, ensure_2d=False, dtype="numeric", ensure_all_finite=True)
        x = x.ravel()

        # Ordenar datos basado en x
        indices_ordenados = np.argsort(x)
        x_ordenado = x[indices_ordenados]
        y_ordenado = y[indices_ordenados]

        clases_unicas = np.unique(y)
        num_muestras = len(x)

        # Identificar posibles puntos de corte (puntos medios)
        valores_unicos_x = np.unique(x_ordenado)
        if len(valores_unicos_x) <= 1:
            self.bordes_intervalo_ = np.array([-np.inf, np.inf])
            self.num_intervalos_ = 1
            self._is_fitted = True
            return self

        limites_potenciales = (valores_unicos_x[:-1] + valores_unicos_x[1:]) / 2.0

        # Inicialización
        limites_actuales = []
        # CAIM inicial (1 intervalo global)
        clases_global, conteos_global = np.unique(y_ordenado, return_counts=True)
        max_global = 0
        if len(conteos_global)>0:
             max_global = np.max(conteos_global)

        caim_actual = (max_global * max_global) / num_muestras if num_muestras > 0 else 0.0
        num_intervalos = 1

        # Búsqueda greedy global
        lista_limites_posibles = list(limites_potenciales)

        while True:
            mejor_nuevo_caim = -1.0
            mejor_candidato_limite = None

            for candidato in lista_limites_posibles:
                limites_temporales = sorted(limites_actuales + [candidato])
                caim_con_candidato = self._calcular_valor_caim(limites_temporales, x_ordenado, y_ordenado, clases_unicas)

                if caim_con_candidato > mejor_nuevo_caim:
                    mejor_nuevo_caim = caim_con_candidato
                    mejor_candidato_limite = candidato

            # Decidir si añadir el mejor candidato
            if mejor_nuevo_caim > caim_actual and mejor_candidato_limite is not None and (num_intervalos + 1) <= len(valores_unicos_x):
                limites_actuales.append(mejor_candidato_limite)
                limites_actuales.sort()
                caim_actual = mejor_nuevo_caim
                num_intervalos += 1
                lista_limites_posibles.remove(mejor_candidato_limite) # Quitar el elegido
            else:
                break # No hay mejora o se alcanzó el límite

        # Crear bordes finales
        self.bordes_intervalo_ = np.unique(np.concatenate(([-np.inf], limites_actuales, [np.inf])))
        self.num_intervalos_ = len(self.bordes_intervalo_) - 1
        self._is_fitted = True
        return self

    def transform(self, x) -> np.ndarray:
        """Aplica la discretización CAIM usando los bordes aprendidos."""
        check_is_fitted(self, 'bordes_intervalo_')
        x_input = check_array(x, ensure_2d=False, dtype="numeric", ensure_all_finite=False)
        if x_input.ndim != 1:
            raise ValueError("La entrada 'x' debe ser unidimensional.")

        x_series = pd.Series(x_input)
        x_discretizado = pd.cut(x_series, bins=self.bordes_intervalo_, labels=False,
                                include_lowest=True, duplicates='drop')

        # Manejo de tipos y NaNs
        x_discretizado = x_discretizado.astype(float)
        if not x_discretizado.isna().any():
            x_discretizado = x_discretizado.astype(int)

        return x_discretizado.values
    


# Hasta ahora no tenemos el Naive Bayes clásico.
class NaiveBayesCategorico(BaseEstimator, ClassifierMixin):
    """
    Clasificador Naive Bayes para características categóricas/discretas.
    """
    def __init__(self):
        self.classes_ = None
        self.log_priors_ = None
        self.log_likelihoods_ = None
        self.n_features_in_ = None
        self.n_categories_per_feature_ = None
        self.class_counts_ = None # Conteos de clase para cálculos

    def fit(self, X, y):
        """
        Ajusta el modelo Naive Bayes Categórico según X, y.

        Args:
            X (array-like, shape (n_samples, n_features)): Datos de entrenamiento.
            y (array-like, shape (n_samples,)): Etiquetas de clase objetivo.

        Returns:
            self: El objeto clasificador ajustado.
        """
        X, y = check_X_y(X, y, dtype=None)
        if not np.issubdtype(X.dtype, np.integer) or (X < 0).any():
             raise ValueError("Los datos de entrada X deben ser enteros no negativos.")

        self.classes_ = unique_labels(y)
        n_samples, self.n_features_in_ = X.shape
        n_classes = len(self.classes_)

        self.class_counts_ = np.zeros(n_classes, dtype=np.float64)
        for i, c in enumerate(self.classes_):
            self.class_counts_[i] = np.sum(y == c)

        priors = self.class_counts_ / n_samples
        with np.errstate(divide='ignore'): # Ignorar división por cero temporalmente si n_samples=0 (validación lo previene)
             self.log_priors_ = np.log(priors, where=(priors > 0), out=np.full_like(priors, -np.inf))

        self.n_categories_per_feature_ = X.max(axis=0) + 1
        max_cats = np.max(self.n_categories_per_feature_) if X.size > 0 else 0
        
        # Inicializar log-likelihoods con -inf
        self.log_likelihoods_ = np.full((n_classes, self.n_features_in_, max_cats), -np.inf)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_samples_c = X_c.shape[0]

            if n_samples_c == 0:
                continue # Likelihoods permanecen -inf

            for j in range(self.n_features_in_):
                n_categories_j = self.n_categories_per_feature_[j]
                counts_v = np.zeros(n_categories_j)
                for v in range(n_categories_j):
                    counts_v[v] = np.sum(X_c[:, j] == v)

                likelihoods = counts_v / n_samples_c
                
                # Calcular logaritmos, manejando probabilidad 0
                with np.errstate(divide='ignore'): # Ignorar log(0)
                     log_lik_j = np.log(likelihoods, where=(likelihoods > 0), out=np.full_like(likelihoods, -np.inf))
                
                self.log_likelihoods_[i, j, :n_categories_j] = log_lik_j
                # Las categorías > n_categories_j permanecen -inf

        self._is_fitted = True
        return self

    def _calculate_log_likelihood(self, X):
        """Calcula la suma de log-prior y log-verosimilitudes."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        if n_features != self.n_features_in_:
            raise ValueError(f"Número de características esperado {self.n_features_in_}, pero se recibieron {n_features}")

        joint_log_likelihood = np.tile(self.log_priors_, (n_samples, 1))

        for i in range(n_samples):
            for j in range(n_features):
                feature_value = X[i, j]
                n_categories_j = self.n_categories_per_feature_[j]

                if feature_value < 0 or feature_value >= n_categories_j:
                     # Categoría inválida o nunca vista para esta característica en fit
                     joint_log_likelihood[i, :] = -np.inf
                     break # No tiene sentido seguir con esta muestra
                else:
                     # Sumar la log-likelihood precalculada (puede ser -inf)
                     log_lik_feature_all_classes = self.log_likelihoods_[:, j, feature_value]
                     joint_log_likelihood[i, :] += log_lik_feature_all_classes
            
            # Si alguna suma resultó en -inf (por ejemplo, debido a un prior -inf), se mantiene

        return joint_log_likelihood

    def predict_log_proba(self, X) -> np.ndarray:
        """Calcula el logaritmo de P(X|Ck)P(Ck) para X."""
        X = check_array(X, dtype=np.integer)

        if not hasattr(self, 'class_counts_') or self.class_counts_ is None:
             raise RuntimeError("El modelo debe ser ajustado (fit) antes de predecir.")

        joint_log_likelihood = self._calculate_log_likelihood(X)
        return joint_log_likelihood

    def predict_proba(self, X) -> np.ndarray:
        """Devuelve las estimaciones de probabilidad P(Ck|X) para X."""
        log_proba_joint = self.predict_log_proba(X)

        # Normalización LogSumExp
        with np.errstate(under='ignore', divide='ignore'):
             max_log_proba = np.max(log_proba_joint, axis=1, keepdims=True)
             # Reemplazar -inf para evitar problemas en exp y log(sum(exp))
             max_log_proba = np.where(np.isinf(max_log_proba), -700.0, max_log_proba)

             log_prob_x = max_log_proba + np.log(np.sum(np.exp(log_proba_joint - max_log_proba), axis=1, keepdims=True))
             log_posterior = log_proba_joint - log_prob_x

        probabilities = np.exp(log_posterior)
        # Manejar NaNs o Inf que puedan surgir si log_prob_x fue -inf
        probabilities[~np.isfinite(probabilities)] = 0.0

        # Re-normalizar por si la suma no es exactamente 1 o fue 0
        prob_sum = np.sum(probabilities, axis=1, keepdims=True)
        probabilities = np.divide(probabilities, prob_sum, where=prob_sum > 0, out=np.zeros_like(probabilities))


        return probabilities

    def predict(self, X) -> np.ndarray:
        """Realiza la clasificación en un array de vectores de prueba X."""
        log_proba_joint = self.predict_log_proba(X)
        # argmax elige el índice del máximo. Si hay varios -inf, elige el primero.
        indices_predichos = np.argmax(log_proba_joint, axis=1)
        return self.classes_[indices_predichos]

    def _more_tags(self):
        # Indica a scikit-learn que este estimador espera características categóricas
        return {'X_types': ['categorical']}
    



if __name__ == "builtins":
    import pandas as pd
    from sklearn.datasets import load_iris, load_wine, load_diabetes
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    iris = load_iris()
    wine = load_wine()
    diabetes = load_diabetes()
    
    # HIPERPARAMEETROS
    # Para el equal width
    k_bins = [2,3,5,10,15]
    
    # Para el ChiMerge
    max_bins = [2,3,5,10,15]
    
    hiperparametros = {
        "BandWidth": k_bins,
        "ChiMerge": max_bins,
        "CAIM": [None]
    }
    
    # Para CAIM no hay hiperpárametros.
    
    # Cargar los datos
    # -------- Iris ---------
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    # ------- Wine ---------
    wine = load_wine()
    X_wine = wine.data
    y_wine = wine.target
    
    # ------- Diabetes ---------
    diabetes = load_diabetes(scaled=False)
    X_diabetes = diabetes.data
    y_diabetes = diabetes.target
    
    clasificador = NaiveBayesCategorico() # NB clásico
    resultados_iris = []
    resultados_wine = []
    resultados_diabetes = []
    for discretizador in hiperparametros.keys():
        for k in hiperparametros[discretizador]:
            if discretizador == "BandWidth":
                discretizador_obj = DiscretizadorBandWidth(n_bins=k)
            elif discretizador == "ChiMerge":
                discretizador_obj = DiscretizadorChiMerge(max_intervalos=k)
            elif discretizador == "CAIM":
                discretizador_obj = DiscretizadorCAIM()
    
            # Inicializar listas para almacenar resultados
            acc_wine = []
            acc_iris = []
            acc_diabetes = []
            mcc_wine = []
            mcc_iris = []
            mcc_diabetes = []
            
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
                # Discretizar features
                n_features = X_train_iris.shape[1]
                for j in range(n_features):
                    try:
                        X_train_iris[:, j] = discretizador_obj.fit_transform(X_train_iris[:, j], y_train_iris)
                    except TypeError:
                        X_train_iris[:, j] = discretizador_obj.fit_transform(X_train_iris[:, j])
                    X_test_iris[:, j] = discretizador_obj.transform(X_test_iris[:, j])
                print(f"Discretizador: {discretizador}, Hiperparámetro: {k}")
                X_train_iris = X_train_iris.astype(int)
                X_test_iris = X_test_iris.astype(int)
                clasificador.fit(X_train_iris, y_train_iris)
                
                y_pred_iris = clasificador.predict(X_test_iris)
                acc_iris.append(accuracy_score(y_test_iris, y_pred_iris))
                mcc_iris.append(matthews_corrcoef(y_test_iris, y_pred_iris))
                
                # ------ Wine ------
                n_features = X_train_wine.shape[1]
                for j in range(n_features):
                    try:
                        X_train_wine[:, j] = discretizador_obj.fit_transform(X_train_wine[:, j], y_train_wine)
                    except TypeError:
                        X_train_wine[:, j] = discretizador_obj.fit_transform(X_train_wine[:, j])
                    X_test_wine[:, j] = discretizador_obj.transform(X_test_wine[:, j])
                X_train_wine = X_train_wine.astype(int)
                X_test_wine = X_test_wine.astype(int)
                clasificador.fit(X_train_wine, y_train_wine)
                
                y_pred_wine = clasificador.predict(X_test_wine)
                acc_wine.append(accuracy_score(y_test_wine, y_pred_wine))
                mcc_wine.append(matthews_corrcoef(y_test_wine, y_pred_wine))
                
                # ------ Diabetes ------
                # El sexo ya es categórico, así que no lo discretizamos.
                # Discretizar features
                n_features = X_train_diabetes.shape[1]
                
                disc_y = DiscretizadorBandWidth(n_bins=10) # Primero la respuesta Y
                disc_y.fit(y_diabetes)
                y_train_diabetes = disc_y.transform(y_train_diabetes)
                y_test_diabetes = disc_y.transform(y_test_diabetes)
                
                for j in range(n_features):
                    if j != 1:
                        try:
                            X_train_diabetes[:, j] = discretizador_obj.fit_transform(X_train_diabetes[:, j], y_train_diabetes)
                        except TypeError:
                            X_train_diabetes[:, j] = discretizador_obj.fit_transform(X_train_diabetes[:, j])
                        X_test_diabetes[:, j] = discretizador_obj.transform(X_test_diabetes[:, j])
                X_train_diabetes = X_train_diabetes.astype(int)
                X_test_diabetes = X_test_diabetes.astype(int)
                clasificador.fit(X_train_diabetes, y_train_diabetes)
                
                y_pred_diabetes = clasificador.predict(X_test_diabetes)
                acc_diabetes.append(accuracy_score(y_test_diabetes, y_pred_diabetes))
                mcc_diabetes.append(matthews_corrcoef(y_test_diabetes, y_pred_diabetes))
            
            resultados_iris.append({
                "Dataset": "Iris",
                "Discretizador": discretizador,
                "Hiperparámetro": k,
                "Accuracy": np.mean(acc_iris),
                "MCC": np.mean(mcc_iris)
            })
            resultados_wine.append({
                "Dataset": "Wine",
                "Discretizador": discretizador,
                "Hiperparámetro": k,
                "Accuracy": np.mean(acc_wine),
                "MCC": np.mean(mcc_wine)
            })
            resultados_diabetes.append({
                "Dataset": "Diabetes",
                "Discretizador": discretizador,
                "Hiperparámetro": k,
                "Accuracy": np.mean(acc_diabetes),
                "MCC": np.mean(mcc_diabetes)
            })
        
    # Convertir resultados a DataFrame
    resultados_iris_df = pd.DataFrame(resultados_iris)
    resultados_wine_df = pd.DataFrame(resultados_wine)
    resultados_diabetes_df = pd.DataFrame(resultados_diabetes)
    resultados = pd.concat([resultados_iris_df, resultados_wine_df, resultados_diabetes_df],
                           ignore_index=True, axis = 0)
    
    # Guardar resultados en CSV
    resultados.to_csv("./p2_resultados.csv", index=False)
    resultados = pd.read_csv("./Tareas/Tarea5/p2_resultados.csv")
    
    # Graficar resultados
    def barplot_hyper(discretizador, metrica, xlabel):
        """Genera un gráfico de barras para los resultados de un discretizador específico."""
        plt.figure(figsize=(6,4))
        sns.barplot(data=resultados[resultados['Discretizador'] == discretizador], 
                    x='Hiperparámetro', y=metrica, hue='Dataset')
        plt.title(f"{discretizador}")
        plt.xlabel(xlabel)
        plt.ylabel(f"{metrica} Promedio")
        plt.yticks(minor=True)
        plt.legend(title='Dataset')
        plt.savefig(f"./p2_resultados_{discretizador}_{metrica}.pdf", bbox_inches='tight')
    
    barplot_hyper("BandWidth", "Accuracy", "Número de Bins (k)")
    barplot_hyper("BandWidth", "MCC", "Número de Bins (k)")
    
    barplot_hyper("ChiMerge", "Accuracy", "Número de Intervalos (k)")
    barplot_hyper("ChiMerge", "MCC", "Número de Intervalos (k)")
    
    # Para CAIM
    
    resultados_caim = resultados[resultados['Discretizador'] == "CAIM"].copy()
    
    resultados_caim = pd.melt(
        resultados_caim,
        id_vars=["Dataset","Discretizador", "Hiperparámetro"],
        value_vars=["Accuracy", "MCC"],
        var_name="Métrica",
        value_name="Valor"
    )
    
    plt.figure(figsize=(6,4))
    sns.barplot(data = resultados_caim, x = "Dataset", y = "Valor", hue = "Métrica")
    plt.title("CAIM")
    plt.xlabel("Dataset")
    plt.ylabel("Valor Promedio Accuracy/MCC")
    plt.legend(title= "Métrica")
    plt.savefig("./Tareas/Tarea5/p2_resultados_Caim.pdf", bbox_inches='tight')
    
        