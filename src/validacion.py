"""Configuración común para comparar modelos con validación cruzada."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "validacion"
SEED = 42
# Todos los modelos usan las mismas particiones para que la comparación sea justa.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"f1_macro": "f1_macro", "accuracy": "accuracy"}


def cargar_train():
    path = ROOT / "data" / "processed" / "train.csv"
    if not path.exists():
        raise FileNotFoundError("Primero ejecuta src/preparar_datos.py.")
    train = pd.read_csv(path, index_col="fila_original")
    return train.drop(columns="Class"), train["Class"]


def buscar_hiperparametros(name, model, grid, X, y):
    search = GridSearchCV(
        model, grid, scoring=SCORING, refit="f1_macro", cv=CV,
        return_train_score=True, n_jobs=-1,
    )
    search.fit(X, y)
    RESULTS.mkdir(parents=True, exist_ok=True)

    columns = ["params", "mean_test_f1_macro", "std_test_f1_macro",
               "mean_train_f1_macro", "mean_test_accuracy", "rank_test_f1_macro"]
    grid_table = pd.DataFrame(search.cv_results_)[columns].sort_values("rank_test_f1_macro")
    grid_table.to_csv(RESULTS / f"{name}_busqueda.csv", index=False)

    best = search.best_index_
    results = search.cv_results_
    summary = {
        "modelo": name,
        "mejores_parametros": {key: _json_value(value) for key, value in search.best_params_.items()},
        "f1_macro_validacion": round(float(results["mean_test_f1_macro"][best]), 4),
        "f1_macro_desviacion": round(float(results["std_test_f1_macro"][best]), 4),
        "f1_macro_entrenamiento": round(float(results["mean_train_f1_macro"][best]), 4),
        "accuracy_validacion": round(float(results["mean_test_accuracy"][best]), 4),
        "combinaciones_probadas": len(grid_table),
    }
    (RESULTS / f"{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(search.best_estimator_, RESULTS / f"{name}.joblib")
    print(f"{name}: F1 macro en validación {summary['f1_macro_validacion']:.4f} "
          f"(± {summary['f1_macro_desviacion']:.4f}) con {summary['mejores_parametros']}")
    return search, summary


def _json_value(value):
    # numpy y None no siempre se guardan bien en JSON.
    if hasattr(value, "item"):
        return value.item()
    return value
