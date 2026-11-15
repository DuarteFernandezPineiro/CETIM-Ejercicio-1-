# Evaluación de YOLO en CODD

## Diseño y selección

La pregunta experimental fue si un detector compacto entrenado con CODD podía localizar y distinguir diez clases de residuos. La conversión preservó las particiones del conjunto. El entrenamiento usó `train`, y la arquitectura, el checkpoint y los umbrales se escogieron con `val`. Las predicciones y métricas finales de `test` se generaron después de fijar esas decisiones. La auditoría de archivos y etiquetas sí incluye las tres particiones. [Auditoría de CODD](01_dataset_analysis.md).

| Decisión | Motivo y alcance |
|---|---|
| Detección por cajas con YOLO nano | Las cajas permiten estudiar presencia y localización. La segmentación exigiría validar los polígonos y medir otra salida. Se partió de pesos COCO y se ajustó el detector para CODD con `freeze=0`, sin congelación adicional solicitada. Ultralytics mantiene fijos los pesos DFL según su implementación. |
| YOLOv8n frente a YOLO11n, a 640 px | Dos modelos de la misma escala permiten contrastar alternativas dentro de los 4 GiB de VRAM. Se mantuvieron semilla 42, batch de entrenamiento 2, AdamW, ajuste coseno y presupuesto de 60 épocas con paciencia 10. Una entrada mayor conservaría más píxeles por objeto, con más coste de cómputo. Comparar a una resolución común acota la pregunta a los dos modelos bajo ese presupuesto. |
| mAP50–95 para selección | Promedia AP por clase y por IoU de 0,50 a 0,95. Recompensa la detección con cajas bien ajustadas y da el mismo peso a cada clase en el promedio. mAP50 facilita leer el rendimiento con un criterio geométrico más permisivo. Precisión, recall y F1 al umbral fijo describen otro uso, una decisión concreta de aceptar una caja. |

| Modelo | mAP50–95 val | mAP50 val | Mejor época por mAP50–95 | Épocas ejecutadas | Tiempo de entrenamiento |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0,7209 | 0,7854 | 55 | 60 | 90,6 min |
| YOLO11n | 0,6958 | 0,7676 | 35 | 45 | 73,8 min |

Se seleccionó **YOLOv8n** por la ventaja observada de **2,51 puntos porcentuales** en mAP50–95 y por superar a YOLO11n en 8 de 10 clases. En `foam` la diferencia fue de +0,0022 AP; en `stone`, de +0,1370. YOLO11n obtuvo mejor AP en `general_w` y `wood`. Los tiempos reflejan la duración real de ejecuciones con distintas cantidades de épocas. Las pérdidas de entrenamiento y validación descienden en conjunto. El cambio en la pérdida alrededor de la época 51 de YOLOv8n coincide con el cierre programado de la augmentación mosaic. Con una semilla por modelo, falta evidencia para atribuir la diferencia a la arquitectura o estimar estabilidad entre ejecuciones. [Comparación y registros por época](../artifacts/experiment_comparison/) · [Curvas de aprendizaje](../artifacts/experiment_comparison/learning_curves.png) · [Curvas de pérdida](../artifacts/experiment_comparison/loss_curves.png).

## Punto operativo fijado en validación

Se usó el `best.pt` de YOLOv8n, resolución 640 y, para el diagnóstico, confianza **0,50**, IoU de NMS **0,50** e IoU de coincidencia **0,50**. En validación se hicieron dos barridos de una variable:

- Con confianza 0,25, bajar el IoU de NMS de 0,70 a 0,50 redujo los falsos positivos de 443 a 407 con recall prácticamente igual (0,7176 a 0,7179). Entre 42 objetos con alto solapamiento, el emparejamiento geométrico con clase correcta pasó de 7 a 6 aciertos. Este subgrupo pequeño muestra el coste posible de una supresión más agresiva. [Barrido NMS](../artifacts/validation_followup/nms_sensitivity.csv).
- Con IoU de NMS 0,70, confianza 0,50 dio el mayor F1 observado en la rejilla (0,7878). Confianza 0,45 quedó cerca (0,7862) con mayor recall (0,6920 frente a 0,6858). El punto combinado se comprobó en validación (F1 0,7889). Optimizar una política operativa requeriría definir primero el coste de omitir o desviar un objeto y evaluar combinaciones con ese objetivo. [Barrido de confianza](../artifacts/validation_followup/confidence_sensitivity.csv).

## Resultados en las particiones de CODD

| Partición | Imágenes | Objetos anotados / puntuados para AP | mAP50 | mAP50–95 | Precisión fija | Recall fijo | F1 fijo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validación | 570 | 3.237 / 3.236 | 0,7854 | 0,7209 | 0,9285 | 0,6858 | 0,7889 |
| Test | 573 | 3.224 / 3.224 | 0,8028 | 0,7442 | 0,9287 | 0,7196 | 0,8109 |

En test hubo 2.320 aciertos, 178 falsos positivos y 904 falsos negativos de clase según el diagnóstico histórico. El mAP50–95 fue 2,33 puntos mayor que en validación. Esto describe estas dos particiones. Su composición difiere, no hay intervalos de confianza y la similitud visual documentada entre escenas limita la independencia por captura. Para estimar transferencia haría falta separar sesiones u objetos físicos antes de entrenar y reservar una evaluación nueva. [Dependencia visual entre particiones](01_dataset_analysis.md#dependencia-visual-entre-particiones) · [JSON de validación](../artifacts/final_validation/yolov8n.pt-img640-val-20261114T091521607805Z/evaluation_summary.json) · [JSON de test](../artifacts/final_test/yolov8n.pt-img640-test-20261114T091955137093Z/evaluation_summary.json).

Como referencia histórica, las tablas 11 y 12 de `Detection Results/Results_BoundingBoxes.pdf`, incluido en la descarga de CODD, muestran mAP50–95 de 0,848 en validación y 0,855 en test para YOLOv8n a 640 px. La diferencia justifica reconciliar la copia local y comprobar particiones, configuración y entrenamiento del benchmark antes de interpretar una comparación directa. [Publicación de CODD](https://doi.org/10.1016/j.wasman.2024.02.017).

### Clases con menor rendimiento

| Clase | Objetos val / test | AP50–95 val / test | Precisión fija test | Recall fijo test |
|---|---:|---:|---:|---:|
| `brick` | 372 / 351 | 0,9139 / 0,9388 | 0,9792 | 0,9402 |
| `concrete` | 364 / 384 | 0,9269 / 0,9146 | 0,9446 | 0,9323 |
| `foam` | 295 / 300 | 0,6573 / 0,7542 | 0,8724 | 0,7067 |
| `general_w` | 288 / 266 | 0,6836 / 0,7013 | 0,8643 | 0,6466 |
| `gypsum_board` | 276 / 317 | 0,8554 / 0,8285 | 0,9617 | 0,8707 |
| `pipes` | 260 / 256 | 0,4520 / 0,5737 | 0,9140 | 0,3320 |
| `plastic` | 306 / 266 | 0,4253 / 0,4516 | 0,7101 | 0,3684 |
| `stone` | 245 / 270 | 0,6149 / 0,6246 | 0,9931 | 0,5333 |
| `tile` | 346 / 375 | 0,9106 / 0,8926 | 0,9513 | 0,8853 |
| `wood` | 485 / 439 | 0,7693 / 0,7618 | 0,9543 | 0,7130 |

- **`pipes`.** En test se recuperaron 85 de 256 objetos, con ocho FP asignados a esa clase. En el barrido de validación a confianza 0,25 y NMS 0,70, solo 45 de 165 tuberías de escenas densas se emparejaron a IoU ≥0,50 admitiendo cualquier clase y 23 conservaron la clase correcta. Esa caída señala problemas tanto de cobertura geométrica como de clasificación. Una escena con 45 objetos contiene un caso omitido y otro bien localizado que se predijo como `plastic`. Se puede investigar si pesan la forma alargada, las coincidencias con otros materiales o la composición de esas escenas mediante una muestra estratificada. [Ejemplos de validación](../artifacts/validation_followup/visual_review/pipes/) · [Estratos por densidad](../artifacts/validation_followup/class_by_density.csv).
- **`plastic`.** Tuvo el menor AP50–95 en ambas particiones. En test hubo 98 TP de 266 etiquetas y 40 FP asignados a `plastic`. La matriz incluyó 15 `plastic → general_w` y ocho `plastic → pipes`. En validación se observa un `plastic` solitario con IoU 0,98 predicho como `general_w`. El caso muestra que la densidad explica solo parte de las dificultades posibles. La heterogeneidad del material y la frontera entre etiquetas son hipótesis que requieren revisión humana con criterios de clase explícitos. [Caso de validación](../artifacts/validation_followup/visual_review/plastic/classification_error_01_other-2185.jpg) · [Matriz de test](../artifacts/final_test/yolov8n.pt-img640-test-20261114T091955137093Z/confusion_matrix_absolute.png).
- **`stone`.** En test se recuperaron 144 de 270 objetos y hubo un FP de esta clase. En el mismo barrido de validación, el emparejamiento geométrico con clase correcta obtuvo 1/129 en escenas densas y 47/47 en escenas de uno a cinco objetos. En una escena densa, una predicción `plastic` se asigna a un `stone` con IoU 0,98. La separación por escena es descriptiva. Puede mezclar escala, contexto, oclusión y composición del subconjunto. [Caso de validación](../artifacts/validation_followup/visual_review/stone/classification_error_01_2961.jpg) · [Estratos conjuntos](../artifacts/validation_followup/joint_class_size_density_overlap.csv).

### Densidad, solapamiento y tamaño

El recall de la tabla se calcula con el diagnóstico al punto fijo y usa los objetos anotados de cada grupo como denominador. El solapamiento es el mayor IoU de cada caja anotada con otra caja anotada en la misma imagen. La oclusión visible requeriría medir qué parte del objeto está tapada.

| Factor | Grupo | TP / objetos val | Recall val | TP / objetos test | Recall test |
|---|---|---:|---:|---:|---:|
| Densidad | 1–5 objetos | 656 / 758 | 0,8654 | 644 / 766 | 0,8407 |
| Densidad | 6–9 objetos | 287 / 385 | 0,7455 | 274 / 351 | 0,7806 |
| Densidad | ≥10 objetos | 1.277 / 2.094 | 0,6098 | 1.402 / 2.107 | 0,6654 |
| Solapamiento | IoU <0,10 | 2.125 / 2.583 | 0,8227 | 2.223 / 2.680 | 0,8295 |
| Solapamiento | 0,10 ≤ IoU <0,50 | 92 / 612 | 0,1503 | 97 / 518 | 0,1873 |
| Solapamiento | IoU ≥0,50 | 3 / 42 | 0,0714 | 0 / 26 | 0,0000 |
| Área relativa | ≤1 % | 1.044 / 1.498 | 0,6969 | 1.157 / 1.609 | 0,7191 |
| Área relativa | >1 % y ≤10 % | 1.099 / 1.640 | 0,6701 | 1.081 / 1.513 | 0,7145 |
| Área relativa | >10 % | 77 / 99 | 0,7778 | 82 / 102 | 0,8039 |

En ambos splits el recall cae en escenas densas y con cajas próximas. NMS puede suprimir una detección válida vecina, y el evaluador puede asignar una predicción a un objeto contiguo. El barrido con leves cambios de recall al variar NMS y la proximidad de los resultados agregados de cajas pequeñas y medianas dejan abierta la contribución de cada factor. Hay solo 26 objetos de test en el grupo de mayor solapamiento. Para elegir una intervención convendría revisar cualitativamente ese tipo de escena y separar efectos por clase, tamaño y densidad. Con una hipótesis concreta, se podría contrastar en validación mayor resolución, inferencia por teselas o un cambio de NMS, midiendo también coste. [Mapas por clase](../artifacts/validation_followup/class_factor_heatmaps.png).

## Cómo leer el evaluador

- **AP nativo.** Ultralytics 8.4.210 calcula el AP con confianza mínima 0,001, IoU de NMS 0,70 y varios IoU de evaluación. Su caché descarta una fila duplicada de `general_w` en `val/gen-1029.txt`. El diagnóstico sobre las etiquetas brutas la incluye. Se publican ambos denominadores para conservar la trazabilidad del experimento.
- **Diagnóstico fijo.** Las predicciones se recorren por confianza. Cada predicción intenta asignarse primero a un GT de la misma clase con IoU ≥0,50. En su defecto, puede consumir otro GT como error de clase o de localización (IoU mínimo 0,10). Por eso una predicción errónea de alta confianza puede ocupar el GT antes de una correcta. En una comprobación posterior sobre las mismas predicciones de validación, emparejar primero dentro de cada clase pasó de **2.220 a 2.230 TP** en nueve imágenes con diferencias (recall de 0,6858 a 0,6889). Esta sensibilidad corresponde a validación. El AP nativo usa su propio emparejamiento. Se conserva el diagnóstico original para reproducir las cifras históricas. Una versión futura puede asignar primero TP por clase y luego categorizar los sobrantes, publicando ambas series en validación antes de sustituir una métrica.
- **Lectura de errores.** Una fila `false_negative` significa que el GT quedó sin asignación en ese emparejamiento. Un error de clase o localización también cuenta como FN de clase y FP de predicción. Al analizar geometría, las columnas de «clase correcta» del barrido y de los estratos usan una asignación por IoU que ignora la clase al emparejar y la comprueba después. Por eso pueden diferir de las columnas TP y recall del diagnóstico. Las muestras visuales son selecciones automáticas por caso. Sirven para inspeccionar ejemplos y formular hipótesis. [Tabla de errores de test](../artifacts/final_test/yolov8n.pt-img640-test-20261114T091955137093Z/error_analysis.csv) · [Muestras visuales](../artifacts/validation_followup/visual_review.csv).
- **Tamaño de lote.** Los artefactos históricos indican `batch=2`. Era el lote efectivo de la evaluación nativa, pero el cargador de listas de imágenes de Ultralytics procesaba la lista completa en el diagnóstico. La implementación actual divide las entradas en lotes explícitos. Al repetir validación, se reprodujeron AP, métricas fijas, estratos y asignaciones de error con el checkpoint histórico. El test histórico se conserva como referencia. La comprobación del tamaño de lote y la prueba del emparejamiento se hicieron sobre validación.
- **Velocidad.** La medición nativa de test fue 9,35 ms/imagen en inferencia y 11,97 ms/imagen al sumar preprocesado e inferencia y postprocesado, en la GTX 1650. El recíproco de 9,35 ms caracteriza solo esa etapa. La pasada diagnóstica histórica incluye lectura, preprocesado, inferencia, transferencia y emparejamiento. Una cinta requeriría medir adquisición, procesamiento y decisión de extremo a extremo, en el hardware objetivo y con percentiles.

## Prioridades y alcance

Se corrigieron las condiciones que afectan a la reproducción o a la separación de datos. El validador interpreta el YAML y comprueba sus valores. El comparador excluye registros que declaran uso de test u otra partición. El evaluador y el barrido limitan el tamaño de lote y contrastan el mapeo de clases del checkpoint. Los hashes por partición identifican qué archivos se auditaron y los registros de entrenamiento permiten reconstruir la comparación.

| Cuestión pendiente | Siguiente comprobación | Motivo para aplazarla |
|---|---|---|
| Sesiones o material compartidos entre splits y discrepancia con la ficha de CODD | Contrastar manifiestos, confirmar grupos de captura y congelar un test separado por grupo o instalación. | Hace falta información de captura o revisión manual y un nuevo entrenamiento con otra partición. Se preserva el experimento publicado. |
| Omisiones y confusiones de `pipes`, `plastic` y `stone` | Revisar muestras de aciertos y fallos, con criterios de clase y estratos por escena. Usar esa revisión para decidir si corregir etiquetas, mejorar las capturas o ajustar el detector. | Los ejemplos actuales distinguen patrones de error y dejan su causa abierta. Corregir etiquetas o reponderar clases sin esa revisión confundiría los efectos. |
| Densidad, solapamiento y escala | Medir por estratos en validación con un cambio cada vez. Añadir más escenas de alto solapamiento. | Los factores coexisten y las celdas extremas son pequeñas. Las pruebas de resolución y teselas consumen GPU y solo se justifican con una hipótesis concreta. |
| Variación por semilla y diferencias con el benchmark | Repetir el baseline con varias semillas sobre el mismo corpus verificado y registrar dispersión y tiempos. | Las dos corridas disponibles apoyan la selección experimental. Estimar variabilidad exige entrenamientos adicionales. |
| Uso en cinta y calidad fuera del catálogo | Especificar costes de error, captura, velocidad y revisión manual. Incorporar escenas nuevas y sin residuos, y medir latencia y recall por clase. | CODD permite estudiar detección en sus imágenes. Faltan datos operativos y un conjunto externo etiquetado. |

Los scripts se ejecutan por ruta y cada paso tiene una salida inspeccionable. El validador genera un informe puntual. Si se editan imágenes, etiquetas o el YAML después de validar, hay que regenerarlo. Asimismo, `predict.py` vuelve a escribir el JSON en un destino existente, donde podrían quedar imágenes de una ejecución anterior. Las instrucciones usan una sola conversión por destino y una carpeta de inferencia nueva por prueba. Si el flujo se automatizase para muchos experimentos o usuarios, convendría ligar la validación al contenido de la exportación, proteger escrituras concurrentes y aislar cada ejecución.

El resultado sostiene el uso de YOLOv8n como punto de partida experimental. Para decidir qué mejorar primero, la evidencia disponible apunta a revisar la independencia de las capturas y a estudiar las omisiones por clase y por escena en un protocolo nuevo. Con esos controles se podría medir si una intervención mejora la detección en las condiciones que interesan a CETIM.
