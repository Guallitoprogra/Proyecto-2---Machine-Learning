# Clasificación de variedades de frijol con medidas de tamaño y forma

Rodrigo Ramírez y Denis Rodríguez

## 1. Objetivo

El objetivo es clasificar semillas de frijol en siete variedades usando solo medidas de tamaño y forma tomadas de imágenes. Comparamos cinco modelos, desde una referencia que siempre predice la misma variedad hasta un random forest. La métrica principal es F1 macro, porque las variedades no tienen la misma cantidad de semillas y queremos que todas pesen igual.

## 2. Datos

Usamos el dataset Dry Bean del UCI Machine Learning Repository (2020, licencia CC BY 4.0, https://doi.org/10.24432/C50S4B). Contiene 13,611 semillas de siete variedades registradas en Turquía: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER y SIRA. Cada semilla tiene 16 medidas numéricas:

- **Tamaño (6):** `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `ConvexArea` y `EquivDiameter`.
- **Forma (10):** `AspectRation`, `Eccentricity`, `Extent`, `Solidity`, `roundness`, `Compactness` y `ShapeFactor1` a `ShapeFactor4`.

Las medidas ya vienen extraídas de las imágenes, así que no trabajamos con las fotos. Conservamos los nombres originales de las columnas, incluidos `AspectRation` y `roundness`.

### Revisión de calidad

- No hay valores faltantes ni infinitos.
- No hay medidas en cero o negativas, ni filas imposibles (por ejemplo, un área convexa menor que el área).
- Hay 68 filas repetidas exactamente. Las eliminamos antes de separar los datos para que una misma semilla no quedara en entrenamiento y en test a la vez. Quedaron 13,543 filas.
- No encontramos medidas idénticas con variedades distintas.

### Separación

Reservamos el 20 % de los datos para test con una separación estratificada por variedad y semilla 42. Quedaron 10,834 semillas para entrenamiento y 2,709 para test. Todo el análisis exploratorio, el preprocesamiento y la selección de modelos se hicieron solo con entrenamiento. El test se usó una sola vez, al final.

## 3. Análisis exploratorio

![Distribución de variedades](figuras/distribucion_variedades.png)

Las clases están desbalanceadas. DERMASON representa el 26.2 % del entrenamiento y BOMBAY solo el 3.9 %, es decir, 6.8 veces menos semillas. Por eso usamos F1 macro y estratificación en todas las particiones.

![Medidas de tamaño por variedad](figuras/tamanos_por_variedad.png)

BOMBAY se distingue casi solo por su tamaño: su área mediana (171,224) es más del doble que la de cualquier otra variedad. DERMASON, SEKER y SIRA son las más pequeñas y se traslapan bastante en tamaño. En forma, SEKER es la más redonda y HOROZ la más alargada. BARBUNYA y CALI tienen tamaños parecidos y se distinguen poco.

Con la regla de 1.5 veces el rango intercuartílico, el 22.4 % de las filas tiene algún valor atípico. Casi todos pertenecen a una variedad concreta: en las medidas de tamaño, entre el 91 % y el 100 % son BOMBAY. Como no son errores de medición, no eliminamos ninguno.

También hay mucha redundancia: 15 pares de medidas tienen una correlación de 0.95 o más (`Area` y `ConvexArea` están prácticamente en 1). Mantuvimos las 16 medidas y dejamos que la regularización y la selección de hiperparámetros controlaran ese efecto.

El detalle completo está en [analisis_exploratorio.md](analisis_exploratorio.md).

## 4. Metodología

### Preprocesamiento

Para la regresión logística y KNN escalamos las medidas con `StandardScaler`. El escalador está dentro de un `Pipeline`, de modo que en cada partición aprende la media y la desviación solo con los datos de entrenamiento de esa partición. El árbol y el random forest no dependen de la escala, así que no se escalaron. No hubo imputación porque no hay faltantes.

### Modelos

| Modelo | Hiperparámetros probados | Combinaciones |
|---|---|---:|
| Baseline mayoritario | Ninguno | 1 |
| Regresión logística | `C` (0.01 a 10) y pesos por clase | 8 |
| KNN | Vecinos (3 a 25), pesos y distancia Manhattan o euclidiana | 20 |
| Árbol de decisión | Criterio, profundidad, semillas por hoja y pesos por clase | 112 |
| Random forest (200 árboles) | Profundidad, semillas por hoja, variables por corte y pesos | 16 |

### Validación

Todos los modelos se evaluaron con la misma validación cruzada estratificada de 5 particiones (semilla 42). Así podemos comparar los resultados partición por partición y no solo el promedio. Los hiperparámetros se eligieron por F1 macro. También guardamos la accuracy y el F1 macro de entrenamiento para revisar el sobreajuste.

### Regla de selección

Definimos la regla antes de ver el test:

1. Tomar el modelo con mayor F1 macro en validación.
2. Considerar empatados los modelos que queden a menos de una desviación estándar de ese valor.
3. Entre los empatados, elegir el de menor diferencia entre entrenamiento y validación y, si sigue habiendo empate, el más simple.
