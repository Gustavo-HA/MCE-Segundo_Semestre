import numpy as np
import pandas as pd
import warnings
from problema3 import random_partition
from sklearn.preprocessing import LabelEncoder

class NaiveBayesMixto():
    def __init__(self, variables_continuas):
        self.variables_continuas = variables_continuas
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Identificar tipos de variables
        self._indice_continuos = [i for i in range(n_features) if i in self.variables_continuas]
        self._indice_categoricos = [i for i in range(n_features) if i not in self.variables_continuas]
        
        n_var_categoricas = len(self._indice_categoricos)
        n_var_continuas = len(self._indice_continuos)
        
        if n_var_categoricas > 0:
            X_cat = X[:, self._indice_categoricos]
            try:
                X_cat = X_cat.astype(int)
            except ValueError:
                raise ValueError("Las variables categóricas deben ser enteros.")
            if (X_cat < 0).any():
                raise ValueError("Las variables categóricas no pueden ser negativas.")
        else:
            X_cat = np.empty((n_samples, 0), dtype=int)
            
        if n_var_continuas > 0:
            X_cont = X[:, self._indice_continuos]
            X_cont = X_cont.astype(float)
        else:
            X_cont = np.empty((n_samples, 0), dtype=float)
        
        # Calcular priors
        self.class_count_ = np.zeros(n_classes, dtype=int)
        for i, c in enumerate(self.classes_):
            self.class_count_[i] = np.sum(y == c)
            
        self.priors_ = self.class_count_ / n_samples
        
        # likelihoods 0
        with np.errstate(divide='ignore'):
            self.class_log_prior_ = np.log(self.priors_, where=(self.class_count_ > 0), out=np.full_like(self.priors_, -np.inf))
        
        # Parte categórica
        if n_var_categoricas > 0:
            self._n_categories_per_cat_feature = X_cat.max(axis=0) + 1 if X_cat.size > 0 else np.array([], dtype=int)
            self._max_categories = np.max(self._n_categories_per_cat_feature) if self._n_categories_per_cat_feature.size > 0 else 0

            # Inicializar log-likelihoods categóricos con -inf
            self.feature_log_prob_categorical_ = np.full((n_classes, n_var_categoricas, self._max_categories), -np.inf)

            for i, c in enumerate(self.classes_):
                X_cat_c = X_cat[y == c]
                n_samples_c = X_cat_c.shape[0]

                if n_samples_c == 0:
                    continue # Likelihoods permanecen -inf

                for k in range(n_var_categoricas): # k es el índice DENTRO de las categóricas
                    n_categories_k = self._n_categories_per_cat_feature[k]
                    counts_v = np.zeros(n_categories_k)
                    unique_vals, counts = np.unique(X_cat_c[:, k], return_counts=True)
                    counts_v[unique_vals] = counts

                    likelihoods = counts_v / n_samples_c

                    # Calcular log-likelihoods, manejar probabilidad 0 -> log(0) = -inf
                    with np.errstate(divide='ignore'):
                        log_prob_k = np.log(likelihoods, where=(likelihoods > 0), out=np.full_like(likelihoods, -np.inf))

                    self.feature_log_prob_categorical_[i, k, :n_categories_k] = log_prob_k
        
        else:
            # No hay características categóricas
            self.feature_log_prob_categorical_ = np.empty((n_classes, 0, 0))
            self._n_categories_per_cat_feature = np.array([], dtype=int)
            self._max_categories = 0
            
        # Parte continua
        if n_var_continuas > 0:
            self.theta_ = np.zeros((n_classes, n_var_continuas))
            self.sigma2_ = np.zeros((n_classes, n_var_continuas)) # Varianza puede ser 0

            for i, c in enumerate(self.classes_):
                X_cont_c = X_cont[y == c]
                n_samples_c = X_cont_c.shape[0]

                if n_samples_c > 0:
                    self.theta_[i, :] = np.mean(X_cont_c, axis=0)
                    # Usar varianza poblacional (ddof=0)
                    self.sigma2_[i, :] = np.var(X_cont_c, axis=0, ddof=0)
                # else: theta y sigma2 permanecen 0.0

            # Advertir si alguna varianza es cero
            if np.any(self.sigma2_ == 0):
                 zero_var_indices = np.where(self.sigma2_ == 0)
                 warnings.warn(f"Se encontraron varianzas cero para algunas combinaciones clase/característica continua "
                               f"(ej: clase {zero_var_indices[0][0]}, índice de caract. continua {zero_var_indices[1][0]}). ",
                               UserWarning)
        else:
            # No hay características continuas
            self.theta_ = np.empty((n_classes, 0))
            self.sigma2_ = np.empty((n_classes, 0))

        self.n_features_in_ = n_features
        return self
    
    
    def predict(self, X) -> np.ndarray:
        # Usamos _joint_log_likelihood porque argmax(log P(X|C)P(C)) es lo mismo que argmax(P(C|X))
        joint_log_likelihood = self._joint_log_likelihood(X)

        # argmax elige el índice del máximo. Si hay varios -inf, elige el primero.
        indices_predichos = np.argmax(joint_log_likelihood, axis=1)
        return self.classes_[indices_predichos]

    def _calculate_gaussian_log_pdf(self, X_cont):
        """Calcula el logaritmo de la PDF Gaussiana para características continuas."""
        n_samples = X_cont.shape[0]
        n_classes = len(self.classes_)
        n_continuous_features = self.theta_.shape[1]
        log_prob_cont = np.zeros((n_samples, n_classes))

        for k in range(n_continuous_features): # k es el índice DENTRO de las continuas
            theta_k = self.theta_[:, k]    # shape (n_classes,)
            sigma2_k = self.sigma2_[:, k]  # shape (n_classes,)
            X_cont_k = X_cont[:, k]        # shape (n_samples,)

            # Prevenir división por cero y log(0) si sigma2 es 0 o negativa
            valid_sigma_mask = sigma2_k > 0
            # Inicializar con -inf donde sigma2 es inválida
            log_pdf_k = np.full((n_samples, n_classes), -np.inf)

            if np.any(valid_sigma_mask):
                # Calcular solo para las clases con varianza válida
                sigma2_k_valid = sigma2_k[valid_sigma_mask]
                theta_k_valid = theta_k[valid_sigma_mask]

                # Usar broadcasting: (n_samples, 1) vs (n_valid_classes,) -> (n_samples, n_valid_classes)
                term1 = -0.5 * np.log(2 * np.pi * sigma2_k_valid)
                term2 = -0.5 * ((X_cont_k[:, np.newaxis] - theta_k_valid)**2 / sigma2_k_valid)
                log_pdf_valid = term1 + term2

                # Asignar los resultados calculados a las columnas correspondientes
                log_pdf_k[:, valid_sigma_mask] = log_pdf_valid

            log_prob_cont += log_pdf_k

        return log_prob_cont

    def _calculate_categorical_log_prob(self, X_cat):
        """Calcula el logaritmo de la probabilidad para características categóricas."""
        n_samples = X_cat.shape[0]
        n_classes = len(self.classes_)
        n_categorical_features = self.feature_log_prob_categorical_.shape[1]
        log_prob_cat = np.zeros((n_samples, n_classes))

        for k in range(n_categorical_features): # k es el índice DENTRO de las categóricas
            values = X_cat[:, k]
            n_categories_k = self._n_categories_per_cat_feature[k]

            log_prob_k_all_classes = np.full((n_samples, n_classes), -np.inf) # Default para valores no vistos/inválidos

            # Máscara para valores válidos (dentro del rango visto en fit)
            valid_mask = (values >= 0) & (values < n_categories_k)

            if np.any(valid_mask):
                valid_values = values[valid_mask]
                log_probs_valid = self.feature_log_prob_categorical_[:, k, :n_categories_k][:, valid_values].T
                log_prob_k_all_classes[valid_mask] = log_probs_valid

            log_prob_cat += log_prob_k_all_classes

        return log_prob_cat


    def _joint_log_likelihood(self, X):
        """Calcula la log-verosimilitud conjunta P(X|C)P(C) = log P(C) + sum(log P(X_i|C))."""

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Número de características esperado {self.n_features_in_}, pero se recibieron {X.shape[1]}")

        n_samples = X.shape[0]
        joint_log_likelihood = np.tile(self.class_log_prior_, (n_samples, 1)).astype(np.float64)

        # Separar y procesar características categóricas
        if len(self._indice_categoricos) > 0:
            X_cat = X[:, self._indice_categoricos]
            try:
                 X_cat = X_cat.astype(int)
            except ValueError:
                raise ValueError("Las características categóricas en los datos de predicción deben poder convertirse a enteros.")
            # Comprobar finitud y no negatividad para categóricas
            if not np.all(np.isfinite(X_cat)) or (X_cat < 0).any():
                 raise ValueError("Las características categóricas en los datos de predicción deben ser enteros finitos no negativos.")

            log_prob_cat = self._calculate_categorical_log_prob(X_cat)
            joint_log_likelihood += log_prob_cat

        # Separar y procesar características continuas
        if len(self._indice_continuos) > 0:
            X_cont = X[:, self._indice_continuos]
            try:
                 X_cont = X_cont.astype(float)
            except ValueError:
                raise ValueError("Las características continuas en los datos de predicción deben poder convertirse a números de punto flotante.")
            # Comprobar finitud para continuas
            if not np.all(np.isfinite(X_cont)):
                 raise ValueError("Las características continuas en los datos de predicción deben ser finitas.")

            log_prob_cont = self._calculate_gaussian_log_pdf(X_cont)
            joint_log_likelihood += log_prob_cont

        return joint_log_likelihood



def encode_categorical_features(X, categorical_indices):
    """Codifica las características categóricas como enteros."""
    X_encoded = np.copy(X)
    for i in categorical_indices:
        X_encoded[:, i] = LabelEncoder().fit_transform(X[:, i])
    return X_encoded

# Problema 4.
def main():
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    
    imp = SimpleImputer(strategy="mean")
    
    print("Cargando datasets...")
    adult = pd.read_csv("./data/adult.data", header=None)
    titanic = pd.read_csv("./data/titanic.csv")
    credit_approval = pd.read_csv("./data/crx.data", header=None)
    
    # X e y de cada dataset
    adult_X = adult[adult.columns[:-1]]
    adult_y = adult[adult.columns[-1]]
    
    titanic_X = titanic.drop(columns=["Survived", "Cabin",
                                      "PassengerId", "Name",
                                      "Ticket"])
    titanic_X["Age"] = imp.fit_transform(titanic_X[["Age"]])
    titanic_y = titanic["Survived"]
    
    ca_X = credit_approval[credit_approval.columns[:-1]]
    ca_y = credit_approval[credit_approval.columns[-1]]
    
    print("Preprocesando datasets...")
    # Tirar registros con NaN
    adult_X.replace("?", np.nan, inplace=True)
    titanic_X.replace("?", np.nan, inplace=True)
    ca_X.replace("?", np.nan, inplace=True)
    
    adult_X = adult_X.dropna()
    titanic_X = titanic_X.dropna()
    ca_X = ca_X.dropna()
    
    adult_y = adult_y.iloc[adult_X.index]
    titanic_y = titanic_y.iloc[titanic_X.index]
    ca_y = ca_y.iloc[ca_X.index]
    
    # Variables continuas de cada dataset
    adult_continuous = [0, 2, 4, 10, 11, 12]
    titanic_continuous = [2, 5]
    credit_approval_continuous = [0,1,4,7,12,13]
    
    # Variables categóricas a encoder de cada dataset
    adult_encoded = [1, 3, 5, 6, 7, 8, 9, 13]
    titanic_encoded = [1,6]   
    credit_approval_encoded = [0,3,4,5,6,8,9,10,11,12]
    
    # Codificar variables categóricas
    adult_X_encoded = encode_categorical_features(adult_X.values, adult_encoded)
    titanic_X_encoded = encode_categorical_features(titanic_X.values, titanic_encoded)
    ca_X_encoded = encode_categorical_features(ca_X.values, credit_approval_encoded)
    
    adult_X_encoded = adult_X_encoded.astype(float)
    titanic_X_encoded = titanic_X_encoded.astype(float)
    ca_X_encoded = ca_X_encoded.astype(float)
    
    adult_y = LabelEncoder().fit_transform(adult_y)
    titanic_y = titanic_y.to_numpy()
    ca_y = LabelEncoder().fit_transform(ca_y)

    print("Experimentando...")
    acc_adult = []
    acc_titanic = []
    acc_ca = []
    mcc_adult = []
    mcc_titanic = []
    mcc_ca = []
    
    repeticiones = 10
    seed = 42
    np.random.seed(seed)
    for _ in range(repeticiones):
        print(f"Repetición {_+1}/{repeticiones}...")
        X_train_adult,  y_train_adult,X_test_adult, y_test_adult = random_partition(adult_X_encoded, adult_y)
        X_train_titanic, y_train_titanic,X_test_titanic,  y_test_titanic = random_partition(titanic_X_encoded, titanic_y)
        X_train_ca, y_train_ca , X_test_ca, y_test_ca = random_partition(ca_X_encoded, ca_y)
        
        # Modelos
        nb_adult = NaiveBayesMixto(adult_continuous).fit(X_train_adult, y_train_adult)
        nb_titanic = NaiveBayesMixto(titanic_continuous).fit(X_train_titanic, y_train_titanic)
        nb_ca = NaiveBayesMixto(credit_approval_continuous).fit(X_train_ca, y_train_ca)
        
        # Predicciones
        y_pred_adult = nb_adult.predict(X_test_adult)
        y_pred_titanic = nb_titanic.predict(X_test_titanic)
        y_pred_ca = nb_ca.predict(X_test_ca)
        
        # Métricas
        acc_adult.append(accuracy_score(y_test_adult, y_pred_adult))
        acc_titanic.append(accuracy_score(y_test_titanic, y_pred_titanic))
        acc_ca.append(accuracy_score(y_test_ca, y_pred_ca))
        
        mcc_adult.append(matthews_corrcoef(y_test_adult, y_pred_adult))
        mcc_titanic.append(matthews_corrcoef(y_test_titanic, y_pred_titanic))
        mcc_ca.append(matthews_corrcoef(y_test_ca, y_pred_ca))
        
    # Resultados
    resultados = pd.DataFrame({
        "Dataset": ["Adult"]*10 + ["Titanic"]*10 + ["Credit Approval"]*10,
        "Accuracy": acc_adult + acc_titanic + acc_ca,
        "MCC": mcc_adult + mcc_titanic + mcc_ca,
        "Rep": [i for i in range(1,11)] * 3
    })
    
    resultados.to_csv("./data/p4_resultados.csv", index=False)
        

if __name__ == "__main__":
    main()