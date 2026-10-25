"""Revisa el sobreajuste de los modelos y elige el modelo final."""

from ast import literal_eval
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from comparar_modelos import ORDER, cargar_resultados
from validacion import RESULTS


def curva_profundidad(name, title):
    path = RESULTS / f"{name}_busqueda.csv"
    if not path.exists():
        return None
    grid = pd.read_csv(path)
    params = grid["params"].apply(literal_eval).apply(pd.Series)
    grid = pd.concat([params, grid.drop(columns="params")], axis=1)
    grid["max_depth"] = grid["max_depth"].fillna(0).astype(int)

    # Se dejan los demás parámetros en su mejor valor para ver solo el efecto de la profundidad.
    winner = grid.loc[grid["mean_test_f1_macro"].idxmax()]
    others = [column for column in params.columns if column != "max_depth"]
    same = (grid[others].fillna("None") == winner[others].fillna("None")).all(axis=1)
    best = grid[same].copy()
    best["etiqueta"] = best["max_depth"].map(lambda depth: "sin límite" if depth == 0 else str(depth))
    best = best.sort_values("max_depth", key=lambda depth: depth.replace(0, 10**6))
    best["brecha"] = best["mean_train_f1_macro"] - best["mean_test_f1_macro"]
    best[["etiqueta", "mean_train_f1_macro", "mean_test_f1_macro", "brecha"]].round(4).to_csv(
        RESULTS / f"{name}_profundidad.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(best["etiqueta"], best["mean_train_f1_macro"], marker="o", label="Entrenamiento")
    ax.plot(best["etiqueta"], best["mean_test_f1_macro"], marker="o", label="Validación")
    ax.set_xlabel("Profundidad máxima")
    ax.set_ylabel("F1 macro")
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_profundidad.png", dpi=150)
    plt.close(fig)
    return best


def elegir_modelo(table):
    best = table["f1_macro_validacion"].idxmax()
    # Diferencias menores que la variación entre particiones no son una ventaja clara.
    limit = table.loc[best, "f1_macro_validacion"] - table.loc[best, "f1_macro_desviacion"]
    candidates = table[table["f1_macro_validacion"] >= limit].copy()
    candidates["brecha"] = candidates["f1_macro_entrenamiento"] - candidates["f1_macro_validacion"]
    candidates["orden"] = [ORDER.index(name) if name in ORDER else len(ORDER) for name in candidates.index]
    chosen = candidates.sort_values(["brecha", "orden"]).index[0]
    return best, chosen, candidates


def main():
    table = cargar_resultados()
    table["brecha"] = table["f1_macro_entrenamiento"] - table["f1_macro_validacion"]
    table[["f1_macro_entrenamiento", "f1_macro_validacion", "brecha"]].round(4).to_csv(
        RESULTS / "sobreajuste.csv"
    )
    print(table[["f1_macro_entrenamiento", "f1_macro_validacion", "brecha"]].round(4).to_string())

    for name, title in [("arbol", "Árbol de decisión"), ("random_forest", "Random forest")]:
        curve = curva_profundidad(name, f"{title}: entrenamiento contra validación")
        if curve is not None:
            row = curve.loc[curve["mean_test_f1_macro"].idxmax()]
            print(f"{name}: mejor validación con profundidad {row['etiqueta']}; "
                  f"sin límite la brecha es {curve['brecha'].iloc[-1]:.4f}.")

    missing = [name for name in ORDER if name not in table.index]
    best, chosen, candidates = elegir_modelo(table)
    selection = {
        "modelo": chosen,
        "mejores_parametros": table.loc[chosen, "mejores_parametros"],
        "f1_macro_validacion": table.loc[chosen, "f1_macro_validacion"],
        "mejor_f1_macro": best,
        "empatados_dentro_de_una_desviacion": candidates.index.tolist(),
        "regla": "Entre los modelos a menos de una desviación del mejor F1 macro, "
                 "se elige el de menor brecha entre entrenamiento y validación.",
        "modelos_sin_resultados": missing,
    }
    (RESULTS / "seleccion.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        print(f"Selección provisional: faltan {', '.join(missing)}.")
    print(f"Modelo elegido: {chosen} (mejor F1 macro: {best}; empatados: {candidates.index.tolist()}).")


if __name__ == "__main__":
    main()
