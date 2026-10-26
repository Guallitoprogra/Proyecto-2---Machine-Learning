# Regresión logística y KNN

Resultados sobre las 10,834 filas de entrenamiento, con cinco particiones estratificadas y semilla 42. El escalado se ajusta dentro de cada partición. El test no se utiliza en estos experimentos.

| Modelo | Combinaciones | F1 macro validación | Desviación | Accuracy | F1 macro entrenamiento |
|---|---:|---:|---:|---:|---:|
| Baseline mayoritario | 1 | 0.0593 | 0.0000 | 0.2619 | 0.0593 |
| Regresión logística | 8 | 0.9375 | 0.0044 | 0.9257 | 0.9386 |
| KNN | 20 | 0.9386 | 0.0040 | 0.9272 | 1.0000 |

La mejor regresión logística usa `C=10` y no aplica pesos por clase. La diferencia entre entrenamiento y validación es pequeña (0.0011). Con las variables escaladas, la búsqueda terminó sin alcanzar el límite de 3,000 iteraciones.

El mejor KNN usa 25 vecinos, distancia euclidiana (`p=2`) y pesos por distancia. Su F1 de entrenamiento llega a 1 porque, al predecir las mismas filas usadas al ajustar, cada fila puede encontrarse a sí misma con distancia cero. Ese resultado no representa su desempeño con semillas nuevas; para eso usamos la validación cruzada.

KNN supera a la regresión logística en tres de las cinco particiones, pero la diferencia media es de solo 0.0011, menor que las desviaciones observadas. No hay una ventaja amplia en esta comparación. Ambos superan al baseline en las cinco particiones.

La elección final debe incluir también el árbol y el random forest, con la regla descrita en `seleccion_modelo.md`. Esta tabla no sustituye la evaluación final en test ni una validación externa.

Para reproducir las búsquedas, ejecutar `src/entrenar_regresion_logistica.py` y `src/entrenar_knn.py`. Las tablas completas, resúmenes por partición y pipelines ajustados quedan en `results/validacion`.
