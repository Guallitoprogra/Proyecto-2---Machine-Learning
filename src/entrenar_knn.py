"""Ajusta KNN con las mismas particiones que los otros modelos."""

from sklearn.neighbors import KNeighborsClassifier

from preprocesamiento import crear_pipeline
from validacion import buscar_hiperparametros, cargar_train


def main():
    X, y = cargar_train()
    modelo = crear_pipeline(KNeighborsClassifier())
    parametros = {
        "modelo__n_neighbors": [3, 5, 9, 15, 25],
        "modelo__weights": ["uniform", "distance"],
        # p=1 suma diferencias absolutas; p=2 usa la distancia euclidiana.
        "modelo__p": [1, 2],
    }
    buscar_hiperparametros("knn", modelo, parametros, X, y)


if __name__ == "__main__":
    main()
