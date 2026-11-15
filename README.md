# CETIM · Ejercicio 1 — detección de residuos de construcción y demolición

Este prototipo localiza diez clases de residuos en imágenes RGB mediante cajas delimitadoras. El objetivo es estudiar qué materiales podría identificar un detector ligero para asistir a su separación en una cinta. El problema se relaciona con Smart2ReDuCe, donde CETIM investiga visión artificial y fotónica para mejorar la separación de residuos de construcción y demolición. [Proyecto Smart2ReDuCe de CETIM](https://cetim.es/cetim-desarrolla-tecnologias-4-0-para-mejorar-el-proceso-de-reciclaje-de-los-residuos-de-construccion-y-demolicion/).

## Decisiones y resultado

- **Datos.** CODD v3 ofrece cajas para diez categorías de residuos de construcción y demolición y proporciona particiones para entrenamiento, validación y test. SortWaste también ofrece detección en escenas industriales densas, pero su taxonomía, centrada en residuos municipales y envases, encaja peor con los materiales de construcción de este ejercicio. El tamaño de CODD permitió auditar las etiquetas y entrenar dos modelos ligeros con el equipo disponible. [CODD v3](https://data.mendeley.com/datasets/wds85kt64j/3) · [SortWaste](https://arxiv.org/abs/2601.02299).
- **Modelo y métrica.** Se compararon YOLOv8n y YOLO11n preentrenados en COCO, a 640 px y con una semilla. Las cajas bastan para estudiar presencia y localización. El tamaño nano permite entrenar con 4 GiB de VRAM. Se escogió YOLOv8n por mAP50–95 en validación (**0,7209 frente a 0,6958**), que combina la curva precisión–recall por clase con varios requisitos de ajuste de las cajas. El recall y la precisión al punto operativo muestran cuántos objetos se omiten y cuántas predicciones son correctas al usar un umbral concreto.
- **Resultado y alcance.** En la partición test de CODD, YOLOv8n obtuvo **0,7442 mAP50–95**. En el diagnóstico histórico a confianza 0,50, NMS IoU 0,50 e IoU de emparejamiento 0,50, recuperó **85/256 `pipes`, 98/266 `plastic` y 144/270 `stone`**. El recall bajó del **84,1 %** en escenas de uno a cinco objetos al **66,5 %** en escenas de diez o más. Se observaron escenas visualmente muy similares entre particiones, de modo que estos resultados describen el rendimiento en el test de CODD y su alcance depende de cómo se separaron las capturas.

[Auditoría de datos y dependencia entre particiones](docs/01_dataset_analysis.md) · [Resultados, evaluador y trabajo futuro](docs/04_results_and_conclusions.md) · [Ejemplos de inferencia](artifacts/prediction_examples/README.md).

## Instalación

Usa Python 3.11. Desde la raíz del repositorio, en PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
$device = "cpu"
```

Con GPU NVIDIA y un controlador compatible con CUDA 12.8, usa `requirements-gpu-cu128.txt` en lugar del perfil CPU y asigna `$device = "0"`. Las mediciones publicadas usaron este perfil con una GTX 1650 de 4 GiB. En Linux x86-64, crea el entorno con `python3.11 -m venv .venv`, actívalo con `source .venv/bin/activate` y usa `device=cpu` o `device=0`. Los comandos de Python son los mismos.

## Revisión rápida

Estas operaciones usan los registros incluidos y funcionan sin CODD ni pesos:

```powershell
python -m unittest discover -s tests -v
python scripts/compare_experiments.py --runs-dir artifacts/experiment_comparison/training_logs --output-dir tmp/comparacion
```

La comparación regenera tablas y curvas a partir de los CSV de épocas, la configuración y los resúmenes de los dos entrenamientos. El aviso de que falta `best.pt` es esperado al usar esos registros exportados. Para inspeccionar una predicción sin ejecutar el modelo, consulta la [escena de validación 2966](artifacts/prediction_examples/annotated/2966.jpg) junto con [sus detecciones en JSON](artifacts/prediction_examples/predictions.json).

## Reproducir el experimento

### 1. Auditar y preparar CODD

Descarga [CODD v3](https://data.mendeley.com/datasets/wds85kt64j/3) y coloca la carpeta descomprimida en la raíz del repositorio. El nombre de carpeta por defecto contiene **dos espacios** antes de `(CODD)`.

```text
Construction and Demolition Waste Object Detection Dataset  (CODD)/
├── training/    # JPG y XML juntos
├── validation/
└── testing/
```

```powershell
python scripts/evaluate_dataset.py --output-dir tmp/auditoria
python scripts/prepare_yolo_dataset.py
python scripts/validate_yolo_dataset.py
```

Compara el nuevo `tmp/auditoria/summary.json` con el [resumen versionado](artifacts/dataset_audit/summary.json). Los SHA-256 por partición permiten detectar cambios en nombres, imágenes o XML. La copia auditada contiene **1.984/570/573 imágenes** y **16.519 objetos** en total. La diferencia con la cifra publicada queda documentada en la auditoría. La conversión crea `artifacts/yolo_dataset/`, registra 42 cajas corregidas y conserva las imágenes y los XML de origen. Si ya existe esa salida, usa `--overwrite` al regenerarla y vuelve a ejecutar el validador. Si el origen tiene otro nombre, pasa `--dataset-dir` al auditor y `--source-dir` al conversor.

### 2. Entrenar

```powershell
python scripts/train_yolo.py --models yolov8n yolo11n --epochs 60 --imgsz 640 --batch 2 --patience 10 --workers 4 --seed 42 --device $device --project runs/reproduccion
python scripts/compare_experiments.py --runs-dir runs/reproduccion --output-dir tmp/comparacion-reentrenada
```

Usa `--models yolov8n` para entrenar solo el detector seleccionado. Ultralytics descarga las inicializaciones COCO si hacen falta. Las ejecuciones publicadas duraron aproximadamente 91 minutos para YOLOv8n y 74 para YOLO11n en la GTX 1650. En CPU el entrenamiento puede tardar más y las métricas y los tiempos pueden variar. Si repites la orden en el mismo destino, usa la ruta de la nueva ejecución indicada por Ultralytics en los pasos siguientes.

### 3. Evaluar

```powershell
$checkpoint = "runs/reproduccion/codd-yolov8n-img640-seed42/weights/best.pt"
python scripts/evaluate_model.py --checkpoint $checkpoint --split val --imgsz 640 --batch 2 --conf 0.50 --nms-iou 0.50 --match-iou 0.50 --device $device --output-dir tmp/validacion
python scripts/analyze_validation_followup.py --checkpoint $checkpoint --imgsz 640 --batch 2 --device $device --output-dir tmp/sensibilidad
```

Para reproducir la evaluación final de un modelo y configuración ya fijados, ejecuta el primer comando con `--split test --confirm-test --output-dir tmp/test`. El comparador y el análisis de sensibilidad corresponden a validación. Las cifras publicadas usan el checkpoint histórico. El informe técnico distingue el protocolo nativo de AP del diagnóstico a umbral fijo y documenta el cambio en el tamaño efectivo del lote diagnóstico.

### 4. Ejecutar inferencia

```powershell
python scripts/predict.py --source artifacts/yolo_dataset/images/val/2966.jpg --checkpoint $checkpoint --imgsz 640 --conf 0.50 --iou 0.50 --device cpu --output-dir artifacts/predictions/demo
```

También puedes pasar una imagen propia o un directorio plano de imágenes. Usa un directorio de salida nuevo para cada prueba. El comando escribe las imágenes anotadas y un JSON con clase, confianza y caja XYXY en píxeles. Si dispones de un checkpoint CODD compatible, puedes empezar aquí con una imagen propia.

## Licencias y atribución

El código usa [GNU AGPL-3.0](LICENSE). El proyecto utiliza Ultralytics 8.4.210. Los checkpoints de entrenamiento quedan fuera del repositorio. [Código de Ultralytics](https://github.com/ultralytics/ultralytics) · [Condiciones de Ultralytics para código y pesos](https://www.ultralytics.com/license).

Los inventarios de cajas y los ejemplos visuales derivan de *Construction and Demolition Waste Object Detection Dataset (CODD)*, versión 3, de Demetris Demetriou, Pavlos Mavromatidis, Michael Petrou y Demetris Nicolaides. Se han extraído metadatos, preparado montajes y superpuesto anotaciones o predicciones según el artefacto. El corpus completo se descarga de la fuente. [Dataset y DOI](https://doi.org/10.17632/wds85kt64j.3) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Artículo original](https://doi.org/10.1016/j.wasman.2024.02.017).

## Uso de IA

Usé ChatGPT y Codex para explorar alternativas, apoyar la implementación y revisión de scripts, contrastar interpretaciones y revisar la documentación. También los utilicé en la auditoría final del evaluador y de la reproducibilidad. Me encargué de organizar y planificar las fases del proyecto y de gestionar el historial en Git. Formulé interpretaciones a partir de los resultados y después las contrasté con ChatGPT.
