# Auditoría del dataset CODD

## Objetivo y alcance

La auditoría inicial comprueba la estructura y la geometría de las etiquetas. Una revisión posterior explora la similitud entre capturas de distintas particiones. [Script de auditoría](../scripts/evaluate_dataset.py) · [Artefactos de auditoría](../artifacts/dataset_audit/) · [Evaluación del modelo](04_results_and_conclusions.md).

## Problema y estructura de los datos

El proyecto propone detectar y clasificar residuos de construcción y demolición a partir de imágenes. Cada anotación XML contiene objetos con una clase y una caja delimitadora. El inventario también reconoce puntos de polígono para casi todos los objetos, por lo que el dataset podría permitir explorar segmentación si se valida esa geometría.

El conjunto auditado contiene **3.127 imágenes de 1.920 × 1.200 píxeles y 16.519 objetos en 10 clases**. Todas las imágenes están emparejadas con un XML por nombre base. No hay imágenes sin objetos anotados según el inventario: el mínimo es un objeto por imagen.

La ficha de CODD v3 describe residuos de una instalación de reciclaje de Chipre y declara 3.129 imágenes y 16.545 objetos. La causa de la diferencia de 2 imágenes y 26 objetos con la copia local sigue abierta. Repetir el experimento requiere comprobar qué copia se está usando. El resumen registra un SHA-256 por partición, calculado sobre los nombres ordenados y el hash de cada imagen y XML. El identificador permite verificar la copia efectiva, incluso si los recuentos coinciden. Reconciliarla con la publicación requiere un manifiesto equivalente de la fuente. [Ficha oficial](https://data.mendeley.com/datasets/wds85kt64j/3) · [Resumen y huellas de la copia auditada](../artifacts/dataset_audit/summary.json).

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

Hay diferencias de composición entre particiones. Por ejemplo, `wood` representa el 11,2 % de los objetos de entrenamiento, el 15,0 % de validación y el 13,6 % de test. `concrete` representa el 14,0 % en entrenamiento y el 11,2 % en validación. Estas diferencias cambian el peso de cada clase en las métricas microagregadas. El desglose por clase también ayuda a interpretar el mAP, que promedia el AP de las clases con el mismo peso.

El gráfico [`class_distribution.png`](../artifacts/dataset_audit/class_distribution.png) muestra las proporciones por partición. La tabla con recuentos de instancias e imágenes que contienen cada clase está en [`class_distribution.csv`](../artifacts/dataset_audit/class_distribution.csv). Como las imágenes pueden contener varios objetos y varias clases, el número de instancias no equivale al número de imágenes independientes.

## Densidad de objetos y geometría de las cajas

La mediana es de 2–3 objetos por imagen, mientras que la media supera los 5. Esto indica una distribución asimétrica: muchas imágenes tienen pocos objetos y una minoría tiene escenas densas. Aproximadamente el 67 % de las imágenes contiene cinco objetos o menos; cerca del 23 % contiene diez o más. El máximo es de 72 objetos en una imagen de test. La figura [`objects_per_image.png`](../artifacts/dataset_audit/objects_per_image.png) hace visible esta cola de escenas densas.

Las cajas también cubren fracciones pequeñas del fotograma: **8.236 de 16.519 (49,9 %) tienen un área inferior al 1 % del área de la imagen**. Las medianas de área por clase son menores para `concrete`, `brick`, `stone` y `tile` que para `wood` y `general_w`. Esta combinación de objetos pequeños y escenas con muchos objetos puede ser difícil para un detector, sobre todo si el redimensionado reduce mucho la resolución. La figura [`bbox_geometry.png`](../artifacts/dataset_audit/bbox_geometry.png) resume anchura, altura y área relativas.

La resolución uniforme simplifica la preparación y la comparación de ejemplos, pero no demuestra que haya diversidad suficiente de cámaras, instalaciones, iluminación o condiciones de captura. Los ejemplos anotados de [`annotated_examples.png`](../artifacts/dataset_audit/annotated_examples.png) son una inspección visual puntual —un ejemplo por partición— y no una muestra representativa de todo el conjunto. Sugieren que al menos esos ejemplos tienen fondos oscuros y una escena de captura controlada; no permiten generalizar esa observación a todas las imágenes.

## Calidad e integridad observadas

El [`summary.json`](../artifacts/dataset_audit/summary.json), el [`report.md`](../artifacts/dataset_audit/report.md) y el [`data_issues.csv`](../artifacts/dataset_audit/data_issues.csv) registran estos hallazgos:

- Las 3.127 imágenes tienen XML emparejado; no se registran XML huérfanos ni anotaciones ausentes.
- Todas las imágenes legibles tienen dimensiones de 1.920 × 1.200 px y el script no detecta desacuerdos entre esas dimensiones y las declaradas en XML.
- El SHA-256 encontró cero grupos de copias binarias entre particiones. Hay 21 nombres base reutilizados entre carpetas, por lo que cada imagen se identifica por partición y nombre.
- Se registran **42 cajas fuera de los límites**, un 0,25 % de los objetos. Treinta y cuatro son de `wood`; 5 aparecen en entrenamiento, 10 en validación y 27 en test. Al revisar las coordenadas del inventario, las 42 exceden un borde en un solo píxel: `xmax = 1921` para un ancho de 1.920 o `ymax = 1201` para una altura de 1.200. Es una corrección localizada y reproducible.
- El inventario reconoce puntos de polígono en 16.515 de 16.519 objetos. Esto acredita que hay datos de polígono detectables, pero la auditoría no comprueba que los vértices estén dentro de la imagen, que formen polígonos válidos o que coincidan con el objeto.

El listado por archivo y objeto está en [`data_issues.csv`](../artifacts/dataset_audit/data_issues.csv); las coordenadas y atributos de las cajas están en [`bbox_inventory.csv`](../artifacts/dataset_audit/bbox_inventory.csv), y los metadatos por imagen, en [`image_inventory.csv`](../artifacts/dataset_audit/image_inventory.csv).

## Dependencia visual entre particiones

El hash exacto identifica archivos idénticos. En una revisión exploratoria posterior se compararon recortes centrales en escala de grises con un hash perceptual de diferencias y se inspeccionaron visualmente pares cercanos. Los siguientes casos tienen SHA-256 distintos. Sus nombres y cajas se pueden comprobar en los archivos originales y en el inventario. [Inventario de cajas](../artifacts/dataset_audit/bbox_inventory.csv).

| Particiones | Imágenes | Observación |
|---|---|---|
| Entrenamiento y validación | `gen-844.jpg` / `gen-845.jpg` | Coinciden las posiciones de dos botellas y una caja de `general_w` tiene las mismas coordenadas en ambos XML. |
| Entrenamiento y test | `gen-935.jpg` / `gen-936.jpg` | Dos objetos conservan posición y aspecto con cajas casi iguales. También aparecen cambios de contenido y más anotaciones en test. |
| Validación y test | `gen-981.jpg` / `gen-982.jpg` | Las dos botellas conservan su orientación y sus cajas cambian muy poco. Se aprecia un cambio en un objeto pequeño de la escena. |

El fondo fijo puede acercar hashes perceptuales de escenas distintas. Los objetos y sus posiciones visibles en estos casos sostienen la hipótesis de capturas relacionadas o reutilización de objetos físicos. El cribado fue ilustrativo. Faltan identificadores de sesión, una estimación de la prevalencia y una medida del efecto sobre AP. La similitud puede hacer que parte del test se parezca al material visto al entrenar y reducir la dificultad de generalizar a escenas nuevas.

Se conservan los splits originales para documentar el experimento ejecutado. Resolver la dependencia requiere validar grupos de capturas con metadatos o revisión visual y rehacer el entrenamiento y la evaluación con grupos separados. Un test de otra sesión o instalación permitiría medir el escenario de generalización buscado. Cambiar ahora la partición sobre la que ya se conocen los resultados produciría un experimento nuevo y exigiría otro test reservado.

## Decisiones que se derivan de los datos

- **Corregir geometría verificable.** El conversor transforma coordenadas VOC inclusivas a bordes de píxel y recorta las 42 cajas que exceden un borde. Conserva los originales y registra los cambios en `bbox_corrections.csv`. El validador comprueba las dimensiones, el mapeo de clases y las rutas de las particiones.
- **Mantener detección por cajas como primera pregunta.** Hay etiquetas estructuradas para las diez clases y las cajas bastan para estudiar presencia y localización. La calidad semántica exige revisar si las cajas y clases describen bien los objetos. La auditoría detecta puntos de polígonos, cuya geometría y etiquetado requerirían una validación propia antes de explorar segmentación.
- **Separar cobertura del corpus y transferencia.** Todas las imágenes contienen objetos anotados, así que el corpus no permite estimar falsas alarmas en una cinta vacía. También faltan escenas etiquetadas de otra instalación que midan cambios de cámara, iluminación o flujo de materiales. Se necesitarían capturas nuevas, con negativos y clases fuera de catálogo, para estas preguntas.
