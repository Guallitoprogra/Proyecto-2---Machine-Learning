"""Evalúa en test el modelo elegido previamente con validación."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import sklearn

from analizar_sobreajuste import elegir_modelo
from comparar_modelos import ORDER, cargar_resultados
from validacion import RESULTS, ROOT


def huella(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproducir", action="store_true",
                        help="Reproduce el experimento fijo en results sin cambiar la entrega")
    args = parser.parse_args()
    output = ROOT / ("results/evaluacion_reproducida" if args.reproducir else "docs/evaluacion")
    metrics_path = output / "metricas.json"
    if metrics_path.exists():
        # No volvemos a consultar el test al ejecutar el comando por accidente.
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"Evaluación ya guardada: {saved['modelo']}, F1 macro={saved['f1_macro']:.4f}.")
        print(f"Resultados: {output}")
        return

    table = cargar_resultados()
    missing = set(ORDER) - set(table.index)
    if missing:
        raise ValueError(f"Faltan modelos por validar: {sorted(missing)}")
    selection_path = RESULTS / "seleccion.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _, chosen, _ = elegir_modelo(table)
    if selection["modelos_sin_resultados"] or selection["modelo"] != chosen:
        raise ValueError("Repite analizar_sobreajuste.py: la selección está incompleta o desactualizada.")

    model_path = RESULTS / f"{chosen}.joblib"
    model = joblib.load(model_path)
    expected = table.loc[chosen, "mejores_parametros"]
    if selection["mejores_parametros"] != expected:
        raise ValueError("Los parámetros de la selección no coinciden con los resultados.")
    if any(model.get_params()[key] != value for key, value in expected.items()):
        raise ValueError("El modelo guardado no coincide con los parámetros elegidos.")

    train_path, test_path = [ROOT / "data/processed" / name for name in ("train.csv", "test.csv")]
    reference_path = ROOT / "docs/evaluacion/metricas.json"
    if args.reproducir and reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if chosen != reference["modelo"] or expected != reference["parametros"]:
            raise ValueError("La reproducción debe conservar el modelo y los parámetros de la entrega.")
        if huella(train_path) != reference["train_sha256"] or huella(test_path) != reference["test_sha256"]:
            raise ValueError("La reproducción debe usar las mismas particiones de la entrega.")
    train = pd.read_csv(train_path, index_col="fila_original")
    test = pd.read_csv(test_path, index_col="fila_original")
    columns = train.columns.drop("Class").tolist()
    if list(test.columns) != list(train.columns) or list(model.feature_names_in_) != columns:
        raise ValueError("Las columnas de los datos y del modelo no coinciden.")
    if not train.index.is_unique or not test.index.is_unique or set(train.index) & set(test.index):
        raise ValueError("Hay índices repetidos o compartidos entre entrenamiento y test.")
    shared = train[columns].merge(test[columns].drop_duplicates(), on=columns)
    if len(shared):
        raise ValueError("Hay medidas repetidas entre entrenamiento y test.")
    if test.isna().any().any() or not np.isfinite(test[columns].to_numpy()).all():
        raise ValueError("El test tiene valores faltantes o infinitos.")

    # La elección ya está cerrada. Solo este modelo predice las semillas del test.
    probabilities = model.predict_proba(test[columns])
    labels = model.classes_
    prediction = labels[probabilities.argmax(axis=1)]
    actual = test["Class"]
    if set(actual) != set(labels):
        raise ValueError("Las clases de test no coinciden con las clases del modelo.")
    report = classification_report(actual, prediction, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, prediction, labels=labels)
    predictions = pd.DataFrame({
        "real": actual, "prediccion": prediction,
        "confianza": probabilities.max(axis=1),
    }, index=test.index)
    predictions["acierto"] = predictions["real"] == predictions["prediccion"]
    errors = predictions.loc[~predictions["acierto"]].join(test[columns])
    pairs = errors.groupby(["real", "prediccion"]).size().rename("errores").sort_values(ascending=False)

    output.mkdir(parents=True, exist_ok=True)
    table[["f1_macro_validacion", "f1_macro_desviacion", "accuracy_validacion",
           "f1_macro_entrenamiento"]].to_csv(output / "validacion.csv")
    (output / "seleccion.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    predictions.to_csv(output / "predicciones.csv", index_label="fila_original")
    errors.to_csv(output / "errores.csv", index_label="fila_original")
    pairs.to_csv(output / "confusiones.csv")
    rows = {label: report[label] for label in [*labels, "macro avg", "weighted avg"]}
    pd.DataFrame(rows).T.to_csv(output / "metricas_por_clase.csv", index_label="clase")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / "matriz_confusion.csv", index_label="real")

    fig, ax = plt.subplots(figsize=(9, 7))
    plot = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(plot, ax=ax, label="Semillas")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Variedad predicha", ylabel="Variedad real",
           title=f"Evaluación final: {chosen}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    metrics = {
        "modelo": chosen, "parametros": expected,
        "fecha": datetime.now(ZoneInfo("America/Guatemala")).isoformat(timespec="seconds"),
        "f1_macro": f1_score(actual, prediction, average="macro"),
        "accuracy": accuracy_score(actual, prediction),
        "filas_test": len(test), "errores": len(errors),
        "f1_macro_validacion": float(table.loc[chosen, "f1_macro_validacion"]),
        "train_sha256": huella(train_path), "test_sha256": huella(test_path),
        "modelo_sha256": huella(model_path), "seleccion_sha256": huella(selection_path),
        "sklearn_version": sklearn.__version__,
        "nota": "Modelo fijado por validación. El test no se usa para ajustar ni elegir modelos.",
    }
    # Este registro marca que la evaluación terminó correctamente.
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{chosen}: F1 macro={metrics['f1_macro']:.4f}, accuracy={metrics['accuracy']:.4f}.")
    print(f"Errores: {len(errors)} de {len(test)} semillas. Resultados: {output}")


if __name__ == "__main__":
    main()
