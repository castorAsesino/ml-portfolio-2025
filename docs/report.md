# Proyecto 1 — CNN (CIFAR-10)
**Autor:** Tu Nombre • **Fecha:** 2025-10-04

## Resumen ejecutivo
Se implementó una **CNN baseline** para clasificar imágenes de CIFAR-10. El pipeline descarga el dataset, entrena el modelo,
y registra métricas y artefactos (curvas, matriz de confusión, ejemplos) junto a un `results/summary.csv` con columnas
**accuracy, F1, loss, epochs, params, tiempo**. Este módulo sirve como pieza demostrable en entrevistas.

## Problema y dataset
- **Tarea:** Clasificación multi-clase (10 clases).
- **Dataset:** CIFAR-10 (32×32 RGB; 50k entrenamiento, 10k test). Carga con `torchvision.datasets.CIFAR10`.

## Metodología
- **Arquitectura:** 3 bloques Conv-BN-ReLU + MaxPool; clasificador MLP con Dropout.
- **Hiperparámetros clave:** batch 128, 3–10 épocas, LR 0.01, weight decay 1e-4, optim SGD con momentum 0.9.
- **Aumentación:** RandomCrop(32, padding=4), HorizontalFlip, ColorJitter leve.
- **Recursos computacionales:** CPU/GPU (CUDA si disponible).

## Resultados
- `results/summary.csv` agrega una fila por corrida con: accuracy, F1, loss, epochs, params, tiempo de entrenamiento.
- Figuras en `results/<run_id>/`: `loss_curve.png`, `acc_curve.png`, `confusion_matrix.png`, `examples_correct.png`, `examples_incorrect.png`.

## Discusión
- La CNN baseline converge de forma estable con *augmentations* moderados. La matriz de confusión revela clases con mayor ambigüedad visual.
- Ajustes de LR/regularización pueden mejorar el balance entre *underfitting* y *overfitting*.

## Lecciones aprendidas y trabajo futuro
- Añadir búsqueda de hiperparámetros, Mixup/CutMix, y *label smoothing*.
- Explorar *early stopping* y *reduce-on-plateau* para entrenamientos más largos.
