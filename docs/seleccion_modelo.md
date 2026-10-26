# Sobreajuste y selección del modelo

Todos los resultados salen de validación cruzada estratificada de 5 particiones sobre `train.csv`, con semilla 42. El test sigue sin usarse. Las tablas se generan con `src/comparar_modelos.py` y `src/analizar_sobreajuste.py` en `results/validacion`.

## Árbol de decisión

Se probaron 112 combinaciones. La mejor usa `entropy`, profundidad máxima 10, al menos 5 semillas por hoja y sin pesos por clase.

| Profundidad | F1 entrenamiento | F1 validación | Brecha |
|---|---:|---:|---:|
| 4 | 0.879 | 0.874 | 0.005 |
| 6 | 0.925 | 0.911 | 0.014 |
| 8 | 0.948 | 0.923 | 0.024 |
| 10 | 0.959 | 0.924 | 0.035 |
| 12 | 0.965 | 0.923 | 0.042 |
| 15 | 0.967 | 0.922 | 0.045 |
| sin límite | 0.968 | 0.923 | 0.045 |

Los demás parámetros se dejaron en su mejor valor. A partir de profundidad 8 la validación ya no mejora, pero el entrenamiento sigue subiendo, así que el árbol empieza a memorizar. Con profundidad 4 o 6 el modelo se queda corto. Entre 8 y 10 la diferencia en validación es de apenas 0.0004, así que en la práctica cualquiera de las dos sirve.

El mínimo de semillas por hoja también ayuda. Un árbol sin límite de profundidad y con una sola semilla por hoja llega a F1 de 1.00 en entrenamiento, pero baja a 0.917 en validación.

## Random forest

Se probaron 16 combinaciones con 200 árboles. La mejor no limita la profundidad, usa la mitad de las variables en cada corte, al menos 4 semillas por hoja y no usa pesos por clase. Obtiene un F1 macro de 0.9365 (± 0.0036) en validación.

La brecha con entrenamiento es de 0.038, parecida a la del árbol. Sin embargo, en el bosque eso preocupa menos, porque al promediar muchos árboles la validación sigue siendo más alta que la de cualquier árbol solo. Limitar la profundidad a 12 casi no cambia nada (0.9357 contra 0.9365). Con una sola semilla por hoja el bosque memoriza el entrenamiento (F1 de 0.99) sin ganar en validación.

Los pesos por clase (`balanced`) no mejoraron ninguno de los dos modelos. Parece que BOMBAY, aunque es la clase con menos ejemplos, es tan distinta por tamaño que no necesita ese ajuste.

## Comparación

| Modelo | F1 validación | Desviación | Accuracy | F1 entrenamiento |
|---|---:|---:|---:|---:|
| Baseline | 0.0593 | 0.0000 | 0.2619 | 0.0593 |
| Regresión logística | 0.9375 | 0.0045 | 0.9258 | 0.9385 |
| KNN | 0.9386 | 0.0040 | 0.9272 | 1.0000 |
| Árbol | 0.9238 | 0.0030 | 0.9120 | 0.9589 |
| Random forest | 0.9365 | 0.0036 | 0.9254 | 0.9744 |

El random forest supera al árbol en las 5 particiones, así que la diferencia no es casualidad de una sola partición. Lo mismo pasa con la regresión logística y KNN frente al árbol.

## Regla de selección

1. Se toma el modelo con mayor F1 macro promedio en validación.
2. Se consideran empatados con él los modelos que queden a menos de una desviación estándar de ese mejor resultado, porque una diferencia así es menor que la variación entre particiones.
3. Entre los empatados se elige el de menor brecha entre entrenamiento y validación. Si la brecha también es igual, se prefiere el modelo más simple.

`src/analizar_sobreajuste.py` aplica esta regla y guarda el resultado en `results/validacion/seleccion.json`. Ese archivo también indica qué modelos faltan, para que no se tome como definitiva una selección hecha sin los cinco. El modelo elegido es el único que se evalúa en el test.

## Resultado de la selección

KNN tiene el mejor promedio (0.9386 ± 0.0040), así que se consideran empatados todos los modelos con 0.9346 o más: KNN, la regresión logística y el random forest. Entre ellos, la regresión logística tiene la menor brecha (0.0010), por lo que se elige **la regresión logística con `C=10` y sin pesos por clase**.

La brecha de KNN (0.061) está inflada: con pesos por distancia, cada semilla de entrenamiento se encuentra a sí misma, así que su F1 de entrenamiento siempre sale 1. Aunque no tomáramos en cuenta ese valor, la regresión logística seguiría siendo la opción más simple dentro del empate.
