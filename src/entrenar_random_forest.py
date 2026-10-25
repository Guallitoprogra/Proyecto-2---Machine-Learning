"""Ajusta los hiperparámetros del random forest."""

from sklearn.ensemble import RandomForestClassifier

from validacion import SEED, buscar_hiperparametros, cargar_train


# Con 200 árboles el resultado ya es estable; se ajusta sobre todo la complejidad de cada árbol.
GRID = {
    "max_depth": [12, None],
    "min_samples_leaf": [1, 4],
    "max_features": ["sqrt", 0.5],
    "class_weight": [None, "balanced"],
}


def main():
    X, y = cargar_train()
    forest = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1)
    buscar_hiperparametros("random_forest", forest, GRID, X, y)


if __name__ == "__main__":
    main()
