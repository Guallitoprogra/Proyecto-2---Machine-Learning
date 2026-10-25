"""Evalúa la referencia que siempre predice la variedad más frecuente."""

from sklearn.dummy import DummyClassifier

from preprocesamiento import crear_pipeline
from validacion import SEED, buscar_hiperparametros, cargar_train


def main():
    X, y = cargar_train()
    # La variedad mayoritaria se elige solo con el entrenamiento de cada partición.
    modelo = crear_pipeline(
        DummyClassifier(strategy="most_frequent", random_state=SEED), escalar=False
    )
    # No hay hiperparámetros que ajustar, pero usamos las mismas cinco particiones.
    buscar_hiperparametros("baseline", modelo, {}, X, y)


if __name__ == "__main__":
    main()
