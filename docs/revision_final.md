# Revisión final

Revisión técnica del proyecto. Las métricas finales pueden comprobarse desde las predicciones guardadas con `src/verificar_evaluacion.py`. La revisión manual de los integrantes se registra por separado.

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

- [x] Las métricas del informe coinciden con las predicciones guardadas y su matriz de confusión.
- [x] Ejecución en orden desde una carpeta limpia con el ZIP oficial: mismas particiones, cinco búsquedas y misma selección; la evaluación guardada se reutilizó sin nuevas predicciones de test.
- [x] Pruebas de bloqueo de selección incompleta y reutilización de resultados (`tests/test_evaluacion.py`).
- [ ] Revisar la presentación con los resultados definitivos: 215 errores, F1 macro 0.9310 y caída de 0.0065 respecto a validación.
- [ ] Ambos integrantes revisaron manualmente el código y registraron qué comprobaron en la sección de uso de herramientas del informe.

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
.\.venv\Scripts\python.exe src/evaluar_test.py
.\.venv\Scripts\python.exe src/verificar_evaluacion.py
```

Si la descarga encuentra el problema de certificado documentado en el README, usar el ZIP oficial con `descargar_datos.py --zip "C:\ruta\dry+bean+dataset.zip"`. No se desactiva la verificación HTTPS.

La evaluación guardada se reutiliza al repetir el comando para no volver a predecir el test. Una reproducción independiente del experimento fijo se puede guardar fuera de la entrega con `evaluar_test.py --reproducir`.
