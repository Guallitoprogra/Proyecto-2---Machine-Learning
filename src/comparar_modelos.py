"""Compara los modelos con los resultados de la validación cruzada."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validacion import RESULTS


# Orden para presentar los modelos, del más simple al más complejo.
ORDER = ["baseline", "regresion_logistica", "knn", "arbol", "random_forest"]


def cargar_resultados():
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "f1_macro_por_particion" in summary:
            rows.append(summary)
    if not rows:
        raise FileNotFoundError("No hay resultados en results/validacion; entrena algún modelo primero.")
    table = pd.DataFrame(rows).set_index("modelo")
    ranked = [name for name in ORDER if name in table.index]
    return table.loc[ranked + [name for name in table.index if name not in ranked]]


def main():
    table = cargar_resultados()
    missing = [name for name in ORDER if name not in table.index]
    if missing:
        print(f"Aún faltan resultados de: {', '.join(missing)}.")

    columns = ["f1_macro_validacion", "f1_macro_desviacion", "accuracy_validacion",
               "f1_macro_entrenamiento", "mejores_parametros"]
    table[columns].to_csv(RESULTS / "comparacion.csv")

    # Como todos usan las mismas particiones, se puede comparar partición por partición.
    folds = pd.DataFrame(table["f1_macro_por_particion"].tolist(), index=table.index,
                         columns=[f"particion_{i + 1}" for i in range(5)])
    folds.to_csv(RESULTS / "comparacion_por_particion.csv")
    best = table["f1_macro_validacion"].idxmax()
    wins = (folds.rsub(folds.loc[best], axis=1) > 0).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    ax.bar(x, table["f1_macro_validacion"], yerr=table["f1_macro_desviacion"],
           color="#7a9e7e", capsize=4)
    for i, value in enumerate(table["f1_macro_validacion"]):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, table.index, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 macro (validación cruzada)")
    ax.set_title("Comparación de modelos en 5 particiones")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparacion.png", dpi=150)
    plt.close(fig)

    print(table[["f1_macro_validacion", "f1_macro_desviacion", "accuracy_validacion"]].to_string())
    for name in table.index.drop(best):
        print(f"{best} supera a {name} en {wins[name]} de 5 particiones.")


if __name__ == "__main__":
    main()
