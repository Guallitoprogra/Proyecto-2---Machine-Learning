# Revisión final

Lista de lo que revisamos antes de entregar. Cada punto se comprobó corriendo los scripts desde un clon limpio del repositorio.

## Datos

- [x] Fuente, licencia y DOI documentados (`README.md`, `data/dataset_metadata.json`).
- [x] El Excel original no se modifica y no se sube; se descarga con `src/descargar_datos.py`.
- [x] Revisión de faltantes, infinitos, tipos, duplicados y etiquetas en conflicto (`data/preparacion.json`).
- [x] Duplicados eliminados antes de separar entrenamiento y test.
- [x] Separación estratificada 80/20 con semilla fija.

## Análisis exploratorio

- [x] Distribución de variedades y desbalance.
- [x] Gráficas de tamaño y forma por variedad.
- [x] Valores atípicos revisados y decisión justificada.
- [x] Correlación entre medidas.
- [x] Conclusiones conectadas con las decisiones del modelado (`docs/analisis_exploratorio.md`).
- [x] Todo hecho solo con entrenamiento.

## Modelado

- [x] Los cinco modelos de la propuesta: baseline, regresión logística, KNN, árbol y random forest.
- [x] Escalado dentro del pipeline, sin fuga de información entre particiones.
- [x] Búsqueda de hiperparámetros con la misma validación cruzada para todos.
- [x] F1 macro como métrica principal; accuracy como complemento.
- [x] Análisis de sobreajuste (`docs/seleccion_modelo.md`).
- [x] Regla de selección definida antes de usar el test.

## Evaluación

- [x] El test se usa una sola vez y solo con el modelo elegido.
- [x] Métricas por variedad y matriz de confusión.
- [x] Análisis de los errores más frecuentes.

## Entregables

- [x] Informe con datos, metodología, resultados y limitaciones (`docs/informe.md`).
- [x] Presentación (`docs/presentacion.pptx`).
- [x] README con instrucciones para reproducir cada paso.
- [x] Dependencias con versiones (`requirements.txt` y `requirements-lock.txt`).

## Consistencia

- [x] Los números del informe, la presentación y los documentos coinciden con lo que generan los scripts.
- [x] Los scripts corren en orden desde un clon limpio sin errores.
- [x] Commits repartidos entre los dos integrantes según el plan de trabajo.

## Orden para reproducir todo

```powershell
.\.venv\Scripts\python.exe src/descargar_datos.py
.\.venv\Scripts\python.exe src/preparar_datos.py
.\.venv\Scripts\python.exe src/explorar_datos.py
.\.venv\Scripts\python.exe src/entrenar_baseline.py
.\.venv\Scripts\python.exe src/entrenar_regresion_logistica.py
.\.venv\Scripts\python.exe src/entrenar_knn.py
.\.venv\Scripts\python.exe src/entrenar_arbol.py
.\.venv\Scripts\python.exe src/entrenar_random_forest.py
.\.venv\Scripts\python.exe src/comparar_modelos.py
.\.venv\Scripts\python.exe src/analizar_sobreajuste.py
```

Después corre la evaluación final en test de Rodrigo.
