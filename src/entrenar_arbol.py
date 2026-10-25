"""Ajusta los hiperparámetros del árbol de decisión."""

from sklearn.tree import DecisionTreeClassifier

from validacion import SEED, buscar_hiperparametros, cargar_train


# Profundidad y tamaño de hoja controlan qué tanto se puede memorizar el árbol.
GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [4, 6, 8, 10, 12, 15, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "class_weight": [None, "balanced"],
}


def main():
    X, y = cargar_train()
    # Los árboles no dependen de la escala, así que no se escalan las variables.
    tree = DecisionTreeClassifier(random_state=SEED)
    buscar_hiperparametros("arbol", tree, GRID, X, y)


if __name__ == "__main__":
    main()
