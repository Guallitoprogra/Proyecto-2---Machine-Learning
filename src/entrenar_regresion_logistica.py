"""Ajusta la regresión logística usando solamente entrenamiento."""

from sklearn.linear_model import LogisticRegression

from preprocesamiento import crear_pipeline
from validacion import SEED, buscar_hiperparametros, cargar_train


def main():
    X, y = cargar_train()
    modelo = crear_pipeline(LogisticRegression(max_iter=3000, random_state=SEED))
    # C controla cuánto se limita el tamaño de los coeficientes.
    parametros = {
        "modelo__C": [0.01, 0.1, 1.0, 10.0],
        "modelo__class_weight": [None, "balanced"],
    }
    buscar_hiperparametros("regresion_logistica", modelo, parametros, X, y)


if __name__ == "__main__":
    main()
