"""Comprueba las métricas guardadas sin volver a predecir el test."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from validacion import ROOT


def main():
    output = ROOT / "docs/evaluacion"
    metrics = json.loads((output / "metricas.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output / "predicciones.csv", index_col="fila_original")
    errors = pd.read_csv(output / "errores.csv", index_col="fila_original")
    matrix = pd.read_csv(output / "matriz_confusion.csv", index_col="real")
    per_class = pd.read_csv(output / "metricas_por_clase.csv", index_col="clase")
    correct = predictions.real == predictions.prediccion

    if not predictions.index.is_unique or len(predictions) != metrics["filas_test"]:
        raise ValueError("Las predicciones tienen índices repetidos o un tamaño incorrecto.")
    if not correct.equals(predictions.acierto):
        raise ValueError("La columna de aciertos no coincide con las etiquetas.")
    if set(errors.index) != set(predictions.index[~correct]) or len(errors) != metrics["errores"]:
        raise ValueError("El listado de errores no coincide con las predicciones.")
    actual = predictions.real
    predicted = predictions.prediccion
    np.testing.assert_allclose(f1_score(actual, predicted, average="macro"), metrics["f1_macro"])
    np.testing.assert_allclose(accuracy_score(actual, predicted), metrics["accuracy"])
    np.testing.assert_array_equal(confusion_matrix(actual, predicted, labels=matrix.index), matrix.to_numpy())
    np.testing.assert_allclose(per_class.loc["macro avg", "f1-score"], metrics["f1_macro"])
    np.testing.assert_array_equal(matrix.sum(axis=1), per_class.loc[matrix.index, "support"])
    pairs = pd.read_csv(output / "confusiones.csv")
    expected = errors.groupby(["real", "prediccion"]).size().sort_index()
    actual_pairs = pairs.set_index(["real", "prediccion"])["errores"].sort_index()
    pd.testing.assert_series_equal(expected, actual_pairs, check_names=False)
    print(f"Verificados {len(predictions)} registros y {len(errors)} errores.")
    print("Métricas, matriz, clases y pares de confusión coinciden con las predicciones guardadas.")


if __name__ == "__main__":
    main()
