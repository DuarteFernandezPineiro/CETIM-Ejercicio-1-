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
