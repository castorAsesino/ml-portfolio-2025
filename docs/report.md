# Proyecto 1 — CNN con Fashion-MNIST  

## Resumen ejecutivo  
Entrené una **red neuronal convolucional (CNN)** para clasificar prendas en el dataset **Fashion-MNIST** (70,000 imágenes en 10 clases).  
El modelo combina capas convolucionales con **BatchNorm** y **Dropout**, entrenado 10 épocas con **Adam (lr=0.001, batch=64)**.  
Alcanzó **90.6% de accuracy** y **90.7% de F1 Score**. Funciona muy bien en categorías fáciles (pantalones, bolsos) y presenta más dificultad en clases similares (camisa vs. camiseta).  

## Problema y dataset  
- **Problema:** clasificar ropa en 10 categorías, incluso en casos visualmente parecidos.  
- **Dataset:** Fashion-MNIST (imágenes 28x28 en escala de grises).  

## Metodología  
- **Arquitectura:** CNN con 4 bloques Conv2D + MaxPooling, BatchNorm y Dropout.  
- **Entrenamiento:** 10 épocas, Adam (lr=0.001), batch=64.  
- **Recursos:** entrenado en CPU estándar.  

## Resultados y discusión  
- **Accuracy en test:** 90.6%  
- **F1 Score:** 90.7%  
- **Fortalezas:** pantalones y bolsos casi perfectos (>98%).  
- **Debilidades:** confusión entre camisas y camisetas por similitud visual.  

## Lecciones aprendidas  
- BatchNorm + Dropout ayudan a mejorar generalización.  
- Accuracy no es suficiente; F1 Score y métricas por clase dan más contexto.  
- La visualización de ejemplos ayuda a identificar problemas temprano.  

## Trabajo futuro  
- Aplicar **data augmentation** (rotaciones, zoom).  
- Ajustar hiperparámetros para mayor precisión.  
- Explorar **transfer learning** con modelos preentrenados.  

# Proyecto 4 — LSTM con IMDB 

## Resumen ejecutivo  
Entrené una **red LSTM bidireccional** para clasificar reseñas de películas (dataset **IMDB**, con ejemplos en 2 clases: positivo/negativo).  
El modelo usa **Embedding + SpatialDropout + LSTM con Dropout**. Con 5 épocas alcanzó **85.7% de accuracy en test** y **86.1% de F1 Score**.  

## Problema y dataset  
- **Problema:** clasificar reseñas de texto como positivas o negativas.  
- **Dataset:** IMDB reviews (ya tokenizado).  

## Metodología  
- **Arquitectura:** Embedding, SpatialDropout1D, Bidirectional LSTM (96 unidades), capa densa sigmoide.  
- **Entrenamiento:** 5 épocas, Adam (lr=0.0005), batch=128.  
- **Recursos:** entrenado en CPU estándar.  

## Resultados y discusión  
- **Accuracy en test:** 85.7%  
- **F1 Score:** 86.1%  
- La matriz de confusión muestra buena precisión en ambas clases, con algo más de falsos negativos.  

## Lecciones aprendidas  
- Regularización con Dropout y EarlyStopping reducen overfitting.  
- F1 es mejor métrica que accuracy para evaluar balance entre positivo y negativo.  

## Trabajo futuro  
- Probar **GRU** o Transformers para mejorar performance.  
- Usar embeddings preentrenados (GloVe, Word2Vec).  
- Extender a clasificación multi-clase o análisis más fino de sentimiento.  

