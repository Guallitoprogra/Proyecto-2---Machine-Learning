"""Revisa los datos originales y reserva el conjunto de prueba."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]
CLASSES = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}
SEED = 42


def main():
    source = ROOT / "data" / "raw" / "Dry_Bean_Dataset.xlsx"
    if not source.exists():
        raise FileNotFoundError("Primero ejecuta src/descargar_datos.py.")
    data = pd.read_excel(source, engine="openpyxl")
    if list(data.columns) != PREDICTORS + ["Class"]:
        raise ValueError("Las columnas no coinciden con el archivo original de Dry Bean.")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in PREDICTORS):
        raise ValueError("Todos los predictores deben ser numéricos.")
    missing = data.isna().sum()
    if missing.sum() or not np.isfinite(data[PREDICTORS].to_numpy()).all():
        raise ValueError("Hay valores faltantes o infinitos; se deben revisar antes de separar.")
    if set(data["Class"].unique()) != CLASSES:
        raise ValueError("Las etiquetas no coinciden con las siete variedades esperadas.")

    # No podemos asignar etiquetas distintas a unas mismas medidas sin revisarlas.
    conflicts = data.groupby(PREDICTORS)["Class"].nunique().gt(1).sum()
    if conflicts:
        raise ValueError("Hay medidas idénticas con etiquetas diferentes.")

    duplicates = int(data.duplicated().sum())
    # Conservamos una copia para que las filas repetidas no crucen al test.
    clean = data.drop_duplicates().copy()
    train, test = train_test_split(
        clean, test_size=0.20, random_state=SEED, stratify=clean["Class"]
    )
    train, test = train.sort_index(), test.sort_index()

    destination = ROOT / "data" / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    # La fila original sirve para rastrear errores; no es una variable del modelo.
    train.to_csv(destination / "train.csv", index_label="fila_original")
    test.to_csv(destination / "test.csv", index_label="fila_original")

    report = {
        "source_file": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "original_rows": len(data),
        "predictors": PREDICTORS,
        "types": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_by_column": {column: int(count) for column, count in missing.items()},
        "duplicate_rows_removed": duplicates,
        "conflicting_labels": int(conflicts),
        "remaining_rows": len(clean),
        "split": {
            "method": "Separación estratificada por Class",
            "random_state": SEED,
            "test_fraction": 0.20,
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "decisions": [
            "Se conservan los nombres originales, incluidos AspectRation y roundness.",
            "Se elimina cada repetición exacta conservando la primera fila.",
            "No se eliminan valores extremos ni se escalan variables en esta etapa.",
            "El análisis exploratorio y la validación usarán solamente train.csv.",
            "test.csv queda reservado para una única evaluación final.",
        ],
    }
    (ROOT / "data" / "preparacion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Revisadas {len(data):,} filas: sin faltantes y con {duplicates} duplicados eliminados.")
    print(f"Entrenamiento: {len(train):,} filas. Test reservado: {len(test):,} filas.")


if __name__ == "__main__":
    main()
