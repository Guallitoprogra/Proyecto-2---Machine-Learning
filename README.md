# Clasificación de variedades de frijol

Proyecto de Rodrigo Ramírez y Denis Rodríguez. Usaremos las medidas de tamaño y forma del dataset Dry Bean de UCI para clasificar siete variedades de frijol.

## Preparar el entorno

Se necesita Python 3.12. Desde la carpeta del proyecto, en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements.txt` lista las dependencias principales. `requirements-lock.txt` guarda las versiones usadas para comprobar la instalación.

## Carpetas

- `data/raw`: archivos originales del dataset.
- `data/processed`: datos preparados para los modelos.
- `src`: scripts de preparación, entrenamiento y evaluación.
- `results`: tablas y gráficas de los experimentos.

La descarga de datos y los scripts se agregarán en los siguientes avances.
