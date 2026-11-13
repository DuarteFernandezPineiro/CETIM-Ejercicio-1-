# Análisis del dataset CODD antes del entrenamiento

## Objetivo y alcance

Este análisis documenta si el dataset elegido parece adecuado para abordar un problema de visión artificial relacionado con la gestión de residuos de construcción y demolición. La pregunta en esta fase no es qué modelo obtiene mejores resultados, sino si los datos, sus etiquetas y sus particiones permiten plantear un experimento de detección defendible.

La auditoría se generó con [`scripts/evaluate_dataset.py`](../scripts/evaluate_dataset.py). Los resultados están en [`artifacts/dataset_audit/`](../artifacts/dataset_audit/). El script inspecciona las imágenes y anotaciones XML de `training`, `validation` y `testing`; resume clases y cajas, busca problemas estructurales y genera figuras exploratorias. **No entrena ni evalúa un modelo.** Por tanto, este documento no presenta métricas de rendimiento predictivo.

## Problema y estructura de los datos

El proyecto propone detectar y clasificar residuos de construcción y demolición a partir de imágenes. Cada anotación XML contiene objetos con una clase y una caja delimitadora. El inventario también reconoce puntos de polígono para casi todos los objetos, por lo que el dataset podría permitir explorar segmentación si se valida esa geometría.

El conjunto auditado contiene **3.127 imágenes de 1.920 × 1.200 píxeles y 16.519 objetos en 10 clases**. Todas las imágenes están emparejadas con un XML por nombre base. No hay imágenes sin objetos anotados según el inventario: el mínimo es un objeto por imagen.

| Partición | Imágenes | Objetos | Fracción de imágenes | Fracción de objetos | Media de objetos por imagen | Mediana | Rango |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entrenamiento | 1.984 | 10.058 | 63,4 % | 60,9 % | 5,07 | 3 | 1–51 |
| Validación | 570 | 3.237 | 18,2 % | 19,6 % | 5,68 | 2 | 1–58 |
| Test | 573 | 3.224 | 18,3 % | 19,5 % | 5,63 | 3 | 1–72 |
| **Total** | **3.127** | **16.519** | **100 %** | **100 %** | **5,28** | — | — |

Las particiones de validación y test tienen un número de imágenes parecido y ambas contienen cerca de una quinta parte de los objetos. Sus medias de objetos por imagen son algo mayores que la de entrenamiento; conviene tener presente esta diferencia al interpretar las métricas.

## Distribución de clases

Las diez clases aparecen en las tres particiones. La distribución global no es uniforme, aunque tampoco hay una clase que domine por sí sola: la más frecuente (`concrete`) reúne 2.155 objetos y la menos frecuente (`pipes`), 1.231, una relación aproximada de 1,75 a 1.

| Clase (etiqueta original) | Entrenamiento | Validación | Test | Total | Fracción del total |
|---|---:|---:|---:|---:|---:|
| `brick` | 1.370 | 372 | 351 | 2.093 | 12,7 % |
| `concrete` | 1.407 | 364 | 384 | 2.155 | 13,0 % |
| `foam` | 746 | 295 | 300 | 1.341 | 8,1 % |
| `general_w` | 742 | 288 | 266 | 1.296 | 7,8 % |
| `gypsum_board` | 1.184 | 276 | 317 | 1.777 | 10,8 % |
| `pipes` | 715 | 260 | 256 | 1.231 | 7,5 % |
| `plastic` | 675 | 306 | 266 | 1.247 | 7,6 % |
| `stone` | 744 | 245 | 270 | 1.259 | 7,6 % |
| `tile` | 1.344 | 346 | 375 | 2.065 | 12,5 % |
| `wood` | 1.131 | 485 | 439 | 2.055 | 12,4 % |

Hay diferencias de composición entre particiones. Por ejemplo, `wood` representa el 11,2 % de los objetos de entrenamiento, el 15,0 % de validación y el 13,6 % de test. `concrete` representa el 14,0 % en entrenamiento y el 11,2 % en validación. Esto no invalida las particiones, pero puede hacer que una métrica agregada o una comparación entre validación y test oculte comportamientos distintos por clase. Se deben reportar métricas por clase además del resultado global.

El gráfico [`class_distribution.png`](../artifacts/dataset_audit/class_distribution.png) muestra las proporciones por partición. La tabla con recuentos de instancias e imágenes que contienen cada clase está en [`class_distribution.csv`](../artifacts/dataset_audit/class_distribution.csv). Como las imágenes pueden contener varios objetos y varias clases, el número de instancias no equivale al número de imágenes independientes.

## Densidad de objetos y geometría de las cajas

La mediana es de 2–3 objetos por imagen, mientras que la media supera los 5. Esto indica una distribución asimétrica: muchas imágenes tienen pocos objetos y una minoría tiene escenas densas. Aproximadamente el 67 % de las imágenes contiene cinco objetos o menos; cerca del 23 % contiene diez o más. El máximo es de 72 objetos en una imagen de test. La figura [`objects_per_image.png`](../artifacts/dataset_audit/objects_per_image.png) hace visible esta cola de escenas densas.

Las cajas también cubren fracciones pequeñas del fotograma: **8.236 de 16.519 (49,9 %) tienen un área inferior al 1 % del área de la imagen**. Las medianas de área por clase son menores para `concrete`, `brick`, `stone` y `tile` que para `wood` y `general_w`. Esta combinación de objetos pequeños y escenas con muchos objetos puede ser difícil para un detector, sobre todo si el redimensionado reduce mucho la resolución. La figura [`bbox_geometry.png`](../artifacts/dataset_audit/bbox_geometry.png) resume anchura, altura y área relativas.

La resolución uniforme simplifica la preparación y la comparación de ejemplos, pero no demuestra que haya diversidad suficiente de cámaras, instalaciones, iluminación o condiciones de captura. Los ejemplos anotados de [`annotated_examples.png`](../artifacts/dataset_audit/annotated_examples.png) son una inspección visual puntual —un ejemplo por partición— y no una muestra representativa de todo el conjunto. Sugieren que al menos esos ejemplos tienen fondos oscuros y una escena de captura controlada; no permiten generalizar esa observación a todas las imágenes.

## Calidad e integridad observadas

El [`summary.json`](../artifacts/dataset_audit/summary.json), el [`report.md`](../artifacts/dataset_audit/report.md) y el [`data_issues.csv`](../artifacts/dataset_audit/data_issues.csv) registran estos hallazgos:

- Las 3.127 imágenes tienen XML emparejado; no se registran XML huérfanos ni anotaciones ausentes.
- Todas las imágenes legibles tienen dimensiones de 1.920 × 1.200 px y el script no detecta desacuerdos entre esas dimensiones y las declaradas en XML.
- No se encontraron grupos de imágenes binariamente idénticas entre particiones. Sí hay 21 nombres base repetidos entre carpetas, pero eso no significa que las imágenes sean iguales: el script compara hashes exactos y encontró cero duplicados. Conviene mantener las particiones separadas y no combinar archivos por nombre base.
- Se registran **42 cajas fuera de los límites**, un 0,25 % de los objetos. Treinta y cuatro son de `wood`; 5 aparecen en entrenamiento, 10 en validación y 27 en test. Al revisar las coordenadas del inventario, las 42 exceden un borde en un solo píxel: `xmax = 1921` para un ancho de 1.920 o `ymax = 1201` para una altura de 1.200. Es una corrección localizada y reproducible.
- El inventario reconoce puntos de polígono en 16.515 de 16.519 objetos. Esto acredita que hay datos de polígono detectables, pero la auditoría no comprueba que los vértices estén dentro de la imagen, que formen polígonos válidos o que coincidan con el objeto.

El listado por archivo y objeto está en [`data_issues.csv`](../artifacts/dataset_audit/data_issues.csv); las coordenadas y atributos de las cajas están en [`bbox_inventory.csv`](../artifacts/dataset_audit/bbox_inventory.csv), y los metadatos por imagen, en [`image_inventory.csv`](../artifacts/dataset_audit/image_inventory.csv).

## Idoneidad y límites de la evidencia

**Juicio provisional: adecuado para un primer experimento de detección supervisada y una prueba de concepto.** Hay miles de imágenes, etiquetas estructuradas, instancias de las diez clases en las tres particiones y no aparecen fallos estructurales generalizados. La calidad de las cajas observada es razonablemente buena, con una incidencia de borde pequeña que puede corregirse en la conversión.

La idoneidad para un uso real todavía no está demostrada. Los artefactos no documentan la procedencia exacta del dataset, su licencia, el protocolo de captura, la independencia entre sesiones o centros, ni la política usada para decidir qué residuo pertenece a cada clase. La auditoría tampoco detecta imágenes casi duplicadas, evalúa la consistencia semántica de las etiquetas o establece si la distribución visual representa el entorno donde se usaría el modelo. Estos puntos condicionan cualquier afirmación de generalización y deben documentarse o verificarse antes de presentar el resultado como aplicable en operación.

La inspección del XML indica que los datos tienen estructura compatible con detección mediante cajas y que existe información de polígono para casi todos los objetos. Sin embargo, este análisis no evalúa máscaras ni permite concluir que la anotación de segmentación esté lista para entrenar un modelo de ese tipo.

## Decisiones técnicas antes del entrenamiento

1. **Preservar el original y preparar una conversión reproducible.** No modificar las imágenes ni XML de origen. Generar una representación de entrenamiento y guardar un registro de archivos transformados y correcciones.
2. **Validar y limitar las cajas.** Al convertir las coordenadas XML al formato que requiera el framework, limitar las 42 coordenadas de borde al intervalo válido `[0, ancho] × [0, alto]`, confirmar que cada caja tenga anchura y altura positivas y conservar un recuento de las correcciones. La conversión debe respetar la convención de coordenadas del formato de origen y destino.
3. **Transformar imagen y etiquetas de forma conjunta.** Aplicar el mismo redimensionado y, si procede, padding a las cajas. Dado que casi la mitad de las cajas ocupa menos del 1 % del fotograma, escoger un tamaño de entrada que no elimine demasiado detalle y revisar visualmente las etiquetas transformadas.
4. **Mantener el protocolo de particiones.** Usar `training` para ajustar el modelo, `validation` para decisiones de configuración y umbrales, y `testing` una sola vez para la evaluación final. No aplicar augmentación a validación o test. Aplicar augmentación solo en entrenamiento y con transformaciones plausibles para el contexto visual.
5. **Empezar con un baseline y evaluar por clase.** Reportar mAP@0.5, mAP@0.5:0.95, precisión y recall por clase, junto con predicciones revisadas visualmente. No decidir de antemano que hacen falta pesos o sobremuestreo: primero comprobar si las diferencias de distribución se traducen en errores sistemáticos por clase.
6. **Revisar el origen y la diversidad.** Registrar fuente y licencia del dataset y, si se conoce, comprobar que las imágenes de particiones distintas no provengan de la misma secuencia o sesión de captura. La ausencia de duplicados exactos no descarta similitud visual ni dependencia entre escenas.

## Conclusión

La auditoría respalda continuar con un baseline de detección después de una conversión controlada de las anotaciones y el tratamiento de las 42 cajas que rebasan el borde un píxel. No justifica todavía una afirmación de rendimiento real ni una selección de arquitectura. La siguiente evidencia necesaria será el resultado en validación y test, desglosado por clase y acompañado de ejemplos de aciertos y errores; también falta documentar la procedencia, licencia y representatividad del conjunto.
