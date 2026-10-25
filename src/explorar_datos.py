"""Análisis exploratorio del conjunto de entrenamiento."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "exploracion"
# Medidas que por definición van de 0 a 1.
RATIOS = ["Eccentricity", "Extent", "Solidity", "roundness",
          "Compactness", "ShapeFactor3", "ShapeFactor4"]
SIZE = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter"]
SHAPE = ["AspectRation", "Eccentricity", "roundness", "Compactness", "Solidity", "Extent"]


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


def cajas_por_variedad(train, columns, name, title):
    order = train.groupby("Class")["Area"].median().sort_values().index
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, column in zip(axes.flat, columns):
        sns.boxplot(data=train, x="Class", y=column, order=order, ax=ax,
                    color="#c9b38a", fliersize=1.5, linewidth=0.8)
        ax.set_title(column)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for ax in axes.flat[len(columns):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=150)
    plt.close(fig)


def valores_atipicos(train):
    rows = []
    any_outlier = pd.Series(False, index=train.index)
    for column in train.columns.drop("Class"):
        q1, q3 = train[column].quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mask = (train[column] < low) | (train[column] > high)
        any_outlier |= mask
        outliers = train[mask]
        # Si casi todos son de una variedad, no son errores: es una semilla distinta.
        top = outliers["Class"].value_counts()
        rows.append({
            "columna": column,
            "atipicos": len(outliers),
            "porcentaje": round(100 * len(outliers) / len(train), 2),
            "variedad_principal": top.index[0] if len(top) else "",
            "porcentaje_de_esa_variedad": round(100 * top.iloc[0] / len(outliers), 1) if len(top) else 0,
        })
    table = pd.DataFrame(rows).set_index("columna")
    table.to_csv(OUTPUT / "valores_atipicos.csv")
    print(f"Filas con al menos un valor atípico (regla IQR): {int(any_outlier.sum()):,} "
          f"({100 * any_outlier.mean():.1f} %).")
    return table


def correlaciones(train):
    corr = train.drop(columns="Class").corr()
    corr.round(3).to_csv(OUTPUT / "correlaciones.csv")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlación entre medidas (entrenamiento)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "correlaciones.png", dpi=150)
    plt.close(fig)

    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strong = pairs[pairs.abs() >= 0.95]
    print(f"Pares de medidas con correlación de 0.95 o más: {len(strong)}.")
    return corr


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = cargar_train()
    print(f"Entrenamiento: {len(train):,} filas y {train.shape[1] - 1} predictores.")
    distribucion_variedades(train)
    revisar_faltantes(train)
    cajas_por_variedad(train, SIZE, "tamanos_por_variedad", "Medidas de tamaño por variedad")
    cajas_por_variedad(train, SHAPE, "formas_por_variedad", "Medidas de forma por variedad")
    valores_atipicos(train)
    correlaciones(train)


if __name__ == "__main__":
    main()
