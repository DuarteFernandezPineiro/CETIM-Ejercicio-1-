# CETIM - Ejercicio 1

## Construction and Demolition Waste Detection

Proyecto de visión artificial para la detección
y clasificación de residuos de construcción
mediante Deep Learning.

## Entorno de entrenamiento

Se recomienda Python 3.11. Instala una sola variante de PyTorch en el entorno
virtual: CPU o GPU NVIDIA. Los archivos fijan versiones compatibles de PyTorch
y TorchVision; `requirements.txt` mantiene las dependencias comunes y la versión
de Ultralytics.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Para entrenamiento con GPU NVIDIA (CUDA 12.8, Windows/Python 3.11):

```powershell
python -m pip install -r requirements-gpu-cu128.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU no disponible')"
```

Para CPU:

```powershell
python -m pip install -r requirements-cpu.txt
```

La build CUDA incluye el runtime de CUDA de PyTorch; no hace falta instalar el
CUDA Toolkit por separado. Sí hace falta un driver NVIDIA compatible. Para
revisar el driver y la memoria de la GPU, ejecuta `nvidia-smi`.

## Auditoría inicial del dataset

Antes de entrenar, se puede generar un informe exploratorio del dataset CODD:

```bash
python -m pip install -r requirements.txt
python scripts/evaluate_dataset.py
```

El informe se guarda en `artifacts/dataset_audit/` e incluye inventario de imágenes y anotaciones, distribución de objetos por clase y división, densidad de objetos por imagen, geometría de cajas, ejemplos visuales y avisos de calidad. El script también detecta archivos sin pareja, XML inválidos, dimensiones incoherentes, cajas fuera de la imagen y duplicados exactos entre divisiones.

La división `validation` se usa para elegir el modelo; `testing` se reserva para la evaluación final. Para detección, conviene informar mAP@0.5, mAP@0.5:0.95 y métricas por clase, además de revisar las imágenes anotadas.

## Conversión a YOLO

Para convertir las anotaciones PASCAL VOC a etiquetas YOLO de detección:

```bash
python scripts/prepare_yolo_dataset.py
```

El script crea `artifacts/yolo_dataset/` con `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`, la lista de clases y un informe de conversión. Convierte las coordenadas VOC a YOLO normalizado y recorta las cajas al tamaño de imagen cuando salen de sus límites. Los originales no se modifican. Para regenerar una salida existente, añade `--overwrite`.

## Validación de la conversión

Después de preparar los datos, valida los pares imagen-etiqueta, los IDs de clase, las cajas normalizadas y los recuentos por división, y genera una hoja de revisión visual:

```bash
python scripts/validate_yolo_dataset.py
```

El informe estructural se guarda en `artifacts/yolo_dataset/validation_report.json` y la vista previa de las cajas convertidas en `artifacts/yolo_dataset/annotation_preview.jpg`. Las cajas corregidas se resaltan en rojo en la vista previa. El script devuelve un código de error si detecta inconsistencias.

## Entrenamiento y comparación de modelos

Con el dataset convertido y validado, se pueden ajustar y comparar modelos Ultralytics con la misma configuración:

```bash
python scripts/train_yolo.py --models yolov8s yolo11s yolo26s --imgsz 640 --batch 2 --seed 42
```

Se admiten los tamaños `n`, `s`, `m`, `l` y `x` de YOLOv8, YOLO11 y YOLO26. Por defecto se entrena solo `yolov8n`. Ultralytics obtiene los checkpoints oficiales `*.pt` preentrenados en COCO; no se usan los pesos incluidos en la carpeta del dataset. El script usa `pretrained=True` y `freeze=0` por defecto, por lo que hace fine-tuning de toda la red. `--freeze N` permite congelar las primeras `N` capas si se quiere probar esa alternativa.

Todos los modelos solicitados reciben los mismos parámetros por defecto: 100 épocas, tamaño 640, batch 16, semilla 42, AdamW, `lr0=0.001`, `weight_decay=0.0005`, warmup de 3 épocas y scheduler coseno. En la GTX 1650 de 4 GiB, sobrescribe el batch por 2 a 640 o 1 a 960 para los modelos `s`. Se puede configurar el optimizador (`SGD`, `MuSGD`, `Adam`, `Adamax`, `AdamW`, `NAdam`, `RAdam`, `RMSProp` o `auto`), `--lr0`, `--lrf`, `--weight-decay`, `--warmup-epochs` y `--scheduler {linear,cosine}`. Los parámetros de augmentación configurables son `--hsv-h`, `--hsv-s`, `--hsv-v`, `--degrees`, `--translate`, `--scale`, `--shear`, `--perspective`, `--flipud`, `--fliplr`, `--mosaic` y `--mixup`. AMP se controla con `--amp`/`--no-amp` y las capas congeladas con `--freeze`.

Si se elige `--batch -1` para que Ultralytics ajuste automáticamente el batch de entrenamiento, la validación independiente usa un batch positivo (1 por defecto). Se puede cambiar con `--val-batch`.

Cada experimento se nombra con arquitectura, resolución y semilla, por ejemplo `codd-yolo11s-img640-seed42`. Durante el entrenamiento se usa `train` para ajustar pesos y `val` para seleccionar/comparar modelos; el script nunca evalúa `test`. `requirements.txt` fija `ultralytics==8.4.210` y el script comprueba esa versión antes de entrenar. La primera ejecución necesita Internet para descargar los checkpoints COCO si aún no están en caché.

Los resultados se guardan en `runs/detect/`: checkpoints `best.pt` y `last.pt`, gráficas de Ultralytics, `experiment_summary.json` por modelo y `model_comparison.json`. El resumen registra configuración efectiva, versión de software, hardware, tiempo y épocas ejecutadas, mejor época, métricas globales y por clase, además de velocidad de validación e inferencia. `--device 0` selecciona la primera GPU y `--device cpu` fuerza CPU. Elige la variante de requirements adecuada al hardware antes de entrenar.

En el equipo donde se preparó este entorno hay una GeForce GTX 1650 con 4 GiB
de VRAM.
Para los modelos `s` y AMP activado, empieza con `--batch 2` a 640 píxeles y
`--batch 1` a 960. Son valores conservadores: el uso real depende del espacio
libre de VRAM, del tamaño del conjunto y de las transformaciones. Si aparece un
error de memoria, reduce el batch o la resolución. Evita `--batch -1` en esta
GPU porque la memoria disponible limita la estimación automática.

La validación durante entrenamiento y la independiente utilizan exclusivamente `val`. Ejecuta una evaluación final sobre `test` solo después de seleccionar el modelo y cerrar la configuración.

Los modelos Ultralytics se distribuyen bajo licencia AGPL-3.0 (con opción Enterprise); revisa sus condiciones antes de reutilizarlos o distribuirlos: [licencia y documentación de Ultralytics](https://docs.ultralytics.com/).
