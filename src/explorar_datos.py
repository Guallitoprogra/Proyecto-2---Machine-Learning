"""Análisis exploratorio del conjunto de entrenamiento."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "exploracion"
# Medidas que por definición van de 0 a 1.
RATIOS = ["Eccentricity", "Extent", "Solidity", "roundness",
          "Compactness", "ShapeFactor3", "ShapeFactor4"]


def cargar_train():
    path = ROOT / "data" / "processed" / "train.csv"
    if not path.exists():
        raise FileNotFoundError("Primero ejecuta src/preparar_datos.py.")
    # Solo miramos entrenamiento para no tomar decisiones viendo el test.
    return pd.read_csv(path, index_col="fila_original")


def distribucion_variedades(train):
    counts = train["Class"].value_counts()
    table = pd.DataFrame({
        "filas": counts,
        "porcentaje": (100 * counts / len(train)).round(2),
    })
    table.index.name = "variedad"
    table.to_csv(OUTPUT / "distribucion_variedades.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(table.index, table["filas"], color="#7a9e7e")
    for i, (rows, pct) in enumerate(zip(table["filas"], table["porcentaje"])):
        ax.text(i, rows + 30, f"{pct:.1f} %", ha="center", fontsize=9)
    ax.set_title("Variedades en el conjunto de entrenamiento")
    ax.set_ylabel("Número de semillas")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "distribucion_variedades.png", dpi=150)
    plt.close(fig)

    ratio = table["filas"].max() / table["filas"].min()
    print(f"Variedades: {table['filas'].idxmax()} es la más común y {table['filas'].idxmin()} la menos común.")
    print(f"La clase mayor tiene {ratio:.1f} veces más semillas que la menor.")
    return table


def revisar_faltantes(train):
    predictors = train.drop(columns="Class")
    table = pd.DataFrame({
        "faltantes": train.isna().sum(),
        "infinitos": np.isinf(predictors).sum().reindex(train.columns, fill_value=0),
        "no_positivos": predictors.le(0).sum().reindex(train.columns, fill_value=0),
        "minimo": predictors.min(),
        "maximo": predictors.max(),
    }).reindex(train.columns).round(4)
    table.index.name = "columna"
    table.to_csv(OUTPUT / "revision_faltantes.csv")

    # Un área convexa menor que el área no tendría sentido físico.
    inconsistent = int((train["ConvexArea"] < train["Area"]).sum())
    outside = int(train[RATIOS].gt(1).sum().sum())
    print(f"Faltantes: {int(table['faltantes'].sum())}. Infinitos: {int(table['infinitos'].sum())}. "
          f"Valores no positivos: {int(table['no_positivos'].sum())}.")
    print(f"Filas con ConvexArea < Area: {inconsistent}. Proporciones mayores que 1: {outside}.")
    return table


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = cargar_train()
    print(f"Entrenamiento: {len(train):,} filas y {train.shape[1] - 1} predictores.")
    distribucion_variedades(train)
    revisar_faltantes(train)


if __name__ == "__main__":
    main()
