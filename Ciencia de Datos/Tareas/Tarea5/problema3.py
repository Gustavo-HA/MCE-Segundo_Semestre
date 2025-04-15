import problema2 as p2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import (
    load_iris,
    load_diabetes,
    load_wine
)
from sklearn.naive_bayes import CategoricalNB, GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
import numpy as np



def random_partition(X, y):
    """
    Particiona aleatoriamente los datos en conjuntos de entrenamiento y prueba.
    """
    
    indices = np.random.choice(len(X), int(len(X)*0.8), replace=False)
    indices_test = np.setdiff1d(np.arange(len(X)), indices)
    X_train, X_test = X[indices], X[indices_test]
    y_train, y_test = y[indices], y[indices_test]
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    return X_train, y_train, X_test, y_test

def comparacion_p1():
    iris = load_iris()
    wine = load_wine()
    diabetes = load_diabetes()
    
    # Inicializar listas para almacenar resultados
    acc_wine = []
    acc_iris = []
    acc_diabetes = []
    mcc_wine = []
    mcc_iris = []
    mcc_diabetes = []
    
    classical_nb = GaussianNB()
    repeticiones = 10
    np.random.seed(42)
    for _ in range(repeticiones):
        X_train_iris, y_train_iris, X_test_iris, y_test_iris = random_partition(iris.data, iris.target)
        X_train_wine, y_train_wine, X_test_wine, y_test_wine = random_partition(wine.data, wine.target)
        X_train_diabetes, y_train_diabetes, X_test_diabetes, y_test_diabetes = random_partition(diabetes.data, diabetes.target)
        
        # Iris 
        classical_nb.fit(X_train_iris, y_train_iris)
        
        y_pred_iris = classical_nb.predict(X_test_iris)
        acc_iris.append(accuracy_score(y_test_iris, y_pred_iris))
        mcc_iris.append(matthews_corrcoef(y_test_iris, y_pred_iris))
        
        # Wine
        classical_nb.fit(X_train_wine, y_train_wine)
        
        y_pred_wine = classical_nb.predict(X_test_wine)
        acc_wine.append(accuracy_score(y_test_wine, y_pred_wine))
        mcc_wine.append(matthews_corrcoef(y_test_wine, y_pred_wine))
        
        # Diabetes
        classical_nb.fit(X_train_diabetes, y_train_diabetes)
        
        y_pred_diabetes = classical_nb.predict(X_test_diabetes)
        acc_diabetes.append(accuracy_score(y_test_diabetes, y_pred_diabetes))
        mcc_diabetes.append(matthews_corrcoef(y_test_diabetes, y_pred_diabetes))
    
    resultados = pd.DataFrame({
        "Dataset": ["Iris"]*10 + ["Wine"]*10 + ["Diabetes"]*10,
        "Modelo": ["SKLearn-GNB"]*30,
        "Accuracy": acc_iris + acc_wine + acc_diabetes,
        "MCC": mcc_iris + mcc_wine + mcc_diabetes,
        "Rep": [i for i in range(1, 11)]*3
    })
    
    p1_resultados = pd.read_csv("./Tareas/Tarea5/p1_resultados.csv")
    
    resultados = pd.concat([resultados, p1_resultados], ignore_index=True)
    
    for dataset in ["Iris", "Wine", "Diabetes"]:
        plt.figure(figsize=(5, 5))
        sns.lineplot(data = resultados[resultados["Dataset"] == dataset], 
                     x = "Rep", y = "Accuracy", hue = "Modelo")
        plt.title(f"Accuracy por repetición - {dataset}")
        plt.savefig(f"./p3_accuracy_{dataset}.pdf", bbox_inches='tight')
    
    for dataset in ["Iris", "Wine", "Diabetes"]:
        plt.figure(figsize=(5, 5))
        sns.lineplot(data = resultados[resultados["Dataset"] == dataset], 
                     x = "Rep", y = "MCC", hue = "Modelo")
        plt.title(f"MCC por repetición - {dataset}")
        plt.savefig(f"./p3_mcc_{dataset}.pdf", bbox_inches='tight')
        
    resultados.to_csv("./Tareas/Tarea5/p3_1_resultados.csv", index=False)


def comparacion_p2():
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
    
    # ------- Wine ---------
    wine = load_wine()
    
    # ------- Diabetes ---------
    diabetes = load_diabetes(scaled=False)
    
    clasificador = CategoricalNB(alpha=1)
    resultados_iris = []
    resultados_wine = []
    resultados_diabetes = []
    for discretizador in hiperparametros.keys():
        for k in hiperparametros[discretizador]:
            if discretizador == "BandWidth":
                discretizador_obj = p2.DiscretizadorBandWidth(n_bins=k)
            elif discretizador == "ChiMerge":
                discretizador_obj = p2.DiscretizadorChiMerge(max_intervalos=k)
            elif discretizador == "CAIM":
                discretizador_obj = p2.DiscretizadorCAIM()
    
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
                # Iris
                X_train_iris, y_train_iris, X_test_iris, y_test_iris = random_partition(iris.data, iris.target)
                # Wine
                X_train_wine, y_train_wine, X_test_wine, y_test_wine = random_partition(wine.data, wine.target)
                # Diabetes
                X_train_diabetes, y_train_diabetes, X_test_diabetes, y_test_diabetes = random_partition(diabetes.data, diabetes.target)
                
                # ------ Iris ------
                # Discretizar features
                n_features = X_train_iris.shape[1]
                for j in range(n_features):
                    try:
                        X_train_iris[:, j] = discretizador_obj.fit_transform(X_train_iris[:, j], y_train_iris)
                    except TypeError:
                        X_train_iris[:, j] = discretizador_obj.fit_transform(X_train_iris[:, j])
                    X_test_iris[:, j] = discretizador_obj.transform(X_test_iris[:, j])
                X_train_iris = X_train_iris.astype(int)
                X_test_iris = X_test_iris.astype(int)
                clasificador.fit(X_train_iris, y_train_iris)
                
                X_test_iris[X_test_iris < 0] = 0 # Asegurarse de que no haya valores negativos
                
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
                
                X_test_wine[X_test_wine < 0] = 0 # Asegurarse de que no haya valores negativos
                
                y_pred_wine = clasificador.predict(X_test_wine)
                acc_wine.append(accuracy_score(y_test_wine, y_pred_wine))
                mcc_wine.append(matthews_corrcoef(y_test_wine, y_pred_wine))
                
                # ------ Diabetes ------
                # El sexo ya es categórico, así que no lo discretizamos.
                # Discretizar features
                n_features = X_train_diabetes.shape[1]
                
                disc_y = p2.DiscretizadorBandWidth(n_bins=10) # Primero la respuesta Y
                disc_y.fit(np.concatenate([y_train_diabetes, y_test_diabetes]))
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
                
                X_test_diabetes[X_test_diabetes < 0] = 0 # Asegurarse de que no haya valores negativos
                
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
        plt.savefig(f"./p3_2_resultados_{discretizador}_{metrica}.pdf", bbox_inches='tight')
    
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
    plt.savefig("./Tareas/Tarea5/p3_2_resultados_Caim.pdf", bbox_inches='tight')
    
    resultados.to_csv("./Tareas/Tarea5/p3_2_resultados.csv", index=False)



def main():
    comparacion_p1()
    comparacion_p2()
    
if __name__ == "__main__":
    main()