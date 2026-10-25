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

## Obtener los datos

```powershell
.\.venv\Scripts\python.exe src/descargar_datos.py
```

El comando descarga el Excel original en `data/raw`. También registra la fecha de obtención, la fuente y las huellas SHA-256 en `data/dataset_metadata.json`. Cada ejecución vuelve a descargar el archivo y actualiza ese registro. El Excel no se sube al repositorio; se obtiene con este comando.

Si Python encuentra un problema con el certificado de la conexión, se puede descargar el ZIP desde el botón Download de UCI y ejecutar:

```powershell
.\.venv\Scripts\python.exe src/descargar_datos.py --zip "C:\ruta\dry+bean+dataset.zip"
```

Fuente: [Dry Bean, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry), publicado en 2020. Contiene 13,611 registros, 16 medidas numéricas y la columna objetivo `Class`, con siete variedades.

Licencia: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Referencia: Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B.

Trabajamos con las medidas ya extraídas de las semillas, sin modificar el archivo original. Los resultados se limitarán a las variedades del dataset.

## Revisar y separar los datos

```powershell
.\.venv\Scripts\python.exe src/preparar_datos.py
```

El script revisa columnas, tipos, faltantes, valores infinitos y duplicados. Guarda el resumen en `data/preparacion.json`. Los nombres originales se conservan, incluidos `AspectRation` y `roundness`.

Se eliminan las repeticiones exactas antes de separar los datos para evitar que una misma fila aparezca en entrenamiento y test. El Excel original permanece intacto. La separación reserva el 20 % para test, mantiene aproximadamente la proporción de cada variedad y usa la semilla 42.

Los archivos `train.csv` y `test.csv` se generan en `data/processed`. Al leerlos con pandas, usar `index_col="fila_original"`: ese índice permite ubicar la semilla en el archivo original y no debe entrar como predictor.

El análisis exploratorio, el preprocesamiento y la selección de modelos se harán con entrenamiento. El escalado se ajustará dentro de cada partición de validación. El test queda reservado para una única evaluación final.

## Análisis exploratorio

```powershell
.\.venv\Scripts\python.exe src/explorar_datos.py
```

Usa solamente `train.csv`. Las tablas y gráficas quedan en `results/exploracion`.

- `distribucion_variedades`: cuántas semillas hay de cada variedad.
- `revision_faltantes`: faltantes, infinitos, valores no positivos y rango de cada columna. También se comprueba que `ConvexArea` no sea menor que `Area` y que las proporciones no pasen de 1.
- `tamanos_por_variedad` y `formas_por_variedad`: cajas de cada medida por variedad, ordenadas de menor a mayor área.
- `valores_atipicos`: atípicos por columna con la regla de 1.5 veces el rango intercuartílico y la variedad a la que pertenecen la mayoría.
- `correlaciones`: correlación entre las 16 medidas.

Las conclusiones y las decisiones que salen de este análisis están en [docs/analisis_exploratorio.md](docs/analisis_exploratorio.md).

## Árbol de decisión y random forest

Los modelos se comparan con `src/validacion.py`: validación cruzada estratificada de 5 particiones, semilla 42 y F1 macro como métrica para elegir hiperparámetros. Todos usan las mismas particiones y solo `train.csv`.

```powershell
.\.venv\Scripts\python.exe src/entrenar_arbol.py
```

Cada modelo deja en `results/validacion`:

- `<modelo>.json`: mejores hiperparámetros, F1 macro en validación y en entrenamiento.
- `<modelo>_busqueda.csv`: todas las combinaciones probadas.
- `<modelo>.joblib`: el modelo con los mejores hiperparámetros ajustado en todo el entrenamiento.

Para el árbol se prueban criterio, profundidad máxima, mínimo de semillas por hoja y pesos por clase. No se escalan las variables porque los cortes del árbol no dependen de la escala.
