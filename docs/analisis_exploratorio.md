# Conclusiones del análisis exploratorio

Todo lo de esta sección sale de `train.csv` (10,834 semillas). El test no se abrió. Las tablas y gráficas se generan con `src/explorar_datos.py` en `results/exploracion`.

## Distribución de variedades

| Variedad | Semillas | % |
|---|---:|---:|
| DERMASON | 2,837 | 26.19 |
| SIRA | 2,109 | 19.47 |
| SEKER | 1,621 | 14.96 |
| HOROZ | 1,488 | 13.73 |
| CALI | 1,304 | 12.04 |
| BARBUNYA | 1,057 | 9.76 |
| BOMBAY | 418 | 3.86 |

Las clases no están balanceadas: DERMASON tiene 6.8 veces más semillas que BOMBAY. Un modelo que siempre diga DERMASON acertaría alrededor del 26 % de las veces, pero su F1 macro sería muy bajo porque las otras seis variedades valdrían cero. Por eso tiene sentido usar F1 macro como métrica principal y mantener la estratificación en la validación cruzada.

## Faltantes y rangos

No hay valores faltantes, infinitos ni medidas en cero o negativas. Tampoco encontramos filas imposibles: en ninguna semilla `ConvexArea` es menor que `Area`, y las medidas que deberían estar entre 0 y 1 (por ejemplo `Solidity`, `roundness` o `ShapeFactor4`) se mantienen en ese rango. No hace falta imputar ni corregir valores.

Lo que sí cambia mucho es la escala. `Area` va de 20,420 a 248,424, mientras que `ShapeFactor2` está entre 0.0006 y 0.0037. Para KNN y regresión logística habrá que escalar, y el escalado se ajustará dentro de cada partición de validación como dice la metodología.

## Tamaño y forma

- BOMBAY se separa casi sola por tamaño. Su mediana de `Area` es 171,224, más del doble que la de cualquier otra variedad.
- En el otro extremo, DERMASON es la más pequeña (mediana 31,874), seguida de SEKER y SIRA. Estas tres se traslapan bastante en las medidas de tamaño.
- La forma ayuda donde el tamaño no alcanza. SEKER es la más redonda (`AspectRation` cerca de 1.24 y `Compactness` de 0.90) y HOROZ es la más alargada (`AspectRation` cerca de 2.04 y `Compactness` de 0.70).
- BARBUNYA y CALI tienen tamaños parecidos. Se distinguen un poco por forma, ya que CALI es más alargada, pero sus cajas siguen traslapándose.
- `Extent` y `Solidity` cambian poco entre variedades, así que probablemente aporten menos que las demás.

Esperamos que los errores se concentren entre DERMASON y SIRA, y entre BARBUNYA y CALI. Lo revisaremos en la matriz de confusión.

## Valores atípicos

Con la regla de 1.5 veces el rango intercuartílico, 2,429 filas (22.4 %) tienen al menos un valor atípico en alguna medida. La mayoría no son errores. En las medidas de tamaño, entre el 91 % y el 100 % de los atípicos son BOMBAY, simplemente porque es mucho más grande. En `AspectRation` casi todos son HOROZ y en `Eccentricity` casi todos son SEKER.

Si quitáramos esas filas estaríamos borrando buena parte de BOMBAY, que ya es la clase con menos ejemplos. Por eso no se elimina ningún atípico. Los casos que no se explican por la variedad (por ejemplo algunos valores bajos de `Solidity` o `roundness`) son pocos y se dejan para ver si aparecen entre los errores del modelo.

## Correlación entre medidas

Hay 15 pares de medidas con correlación de 0.95 o más. `Area` y `ConvexArea` están prácticamente en 1.00, igual que `Compactness` y `ShapeFactor3`. Varias medidas de tamaño (`Area`, `Perimeter`, `EquivDiameter`, `ConvexArea`) repiten casi la misma información.

Esto no afecta mucho al árbol ni al random forest, pero sí puede hacer inestables los coeficientes de la regresión logística y darle más peso al tamaño en las distancias de KNN. Por ahora mantenemos las 16 medidas, y la regularización de la regresión logística se elegirá en la validación cruzada.

## Decisiones para el modelado

1. Usar F1 macro y validación cruzada estratificada.
2. Escalar las variables para regresión logística y KNN dentro de cada partición.
3. No eliminar atípicos ni imputar valores.
4. Conservar las 16 medidas y revisar el efecto de la correlación en los coeficientes.
5. Poner atención a DERMASON contra SIRA y BARBUNYA contra CALI al analizar errores.
