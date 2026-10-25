"""Prepara las medidas dentro de cada partición de entrenamiento."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def crear_pipeline(modelo, escalar=True):
    # El escalador se ajusta junto al modelo, nunca antes de la validación.
    # El análisis no encontró faltantes, así que no necesitamos imputar.
    return Pipeline([
        ("escalado", StandardScaler() if escalar else "passthrough"),
        ("modelo", modelo),
    ])
