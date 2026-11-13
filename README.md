# CETIM - Ejercicio 1

## Construction and Demolition Waste Detection

Proyecto de visión artificial para la detección
y clasificación de residuos de construcción
mediante Deep Learning.

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
