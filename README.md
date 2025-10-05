# Proyecto 1 — CNN con Fashion-MNIST  

## Resumen ejecutivo  
Entrené una CNN para clasificar ropa con el dataset **Fashion-MNIST**.  
Con 10 épocas logré **92.2% de accuracy** y **92.2% de F1 Score**. El modelo anda muy bien en clases fáciles como pantalones y bolsos, pero se complica un poco con camisas y camisetas.  

## Problema y dataset  
- **Problema:** reconocer 10 tipos de prendas.  
- **Dataset:** Fashion-MNIST (70k imágenes 28x28 en escala de grises).  

## Metodología  
- **Arquitectura:** CNN con 4 capas conv + BatchNorm y Dropout.  
- **Entrenamiento:** Adam, lr=0.001, batch=64, 10 épocas.  
- **Recursos:** corrido en CPU común.  

## Resultados  
- Accuracy: **92.2%**  
- F1 Score: **92.2%**  
- Tiempo: ~1242s  
- Parámetros: ~470k  

Fortalezas: pantalón, bolso y sandalia casi perfectos.  
Debilidades: confusión entre camisa y camiseta.  

## Lecciones aprendidas  
- BatchNorm + Dropout ayudan bastante.  
- Ver métricas por clase muestra dónde se equivoca.  

## Trabajo futuro  
- Probar **data augmentation**.  
- Ajustar hiperparámetros.  
- Usar **modelos preentrenados** para comparar.  


## 📌 Resumen ejecutivo  
Entrené un **DCGAN** para generar rostros usando el dataset **CelebA**. El modelo tiene un **generator** y un **discriminator** entrenados con imágenes reales de 64x64.  
Después de 5 épocas, las imágenes generadas todavía no son muy nítidas, pero muestran formas de caras.  

---

## 📝 Problema y dataset  
- **Problema:** generar imágenes sintéticas de rostros que se parezcan a los reales.  
- **Dataset:** CelebA (caras de famosos, recortadas a 64x64).  

---

## ⚙️ Metodología  
- **Arquitectura:**  
  - Generator: 3.57M parámetros  
  - Discriminator: 2.76M parámetros  
  - Latent vector z = 100  
- **Entrenamiento:**  
  - Épocas: 5  
  - Batch size: 64  
  - Learning rate: 0.0002  
- **Recursos:** entrenado en CPU estándar (tardó ~1725s).  

---

## 📊 Resultados y discusión  
- **Pérdida Generator (G):** 7.94  
- **Pérdida Discriminator (D):** 0.60  
- **Tiempo total:** ~1726s  

**Observaciones:**  
- El modelo aprendió formas básicas de caras.  
- Aún le falta nitidez y detalle.  
- Se necesita más entrenamiento y tal vez más filtros.  

📷 Ejemplo: comparación entre imágenes reales y generadas (ver `comparison_final.png`).  

---

## 📚 Lecciones aprendidas  
- Entrenar GANs es más lento y menos estable que CNNs.  
- El balance entre G y D es clave (si uno gana mucho, el otro falla).  
- Las métricas clásicas (accuracy, F1) no sirven: se usa la pérdida y visualización.  

---

## 🚀 Trabajo futuro  
- Entrenar más épocas.  
- Usar GPU para mejorar velocidad.  
- Probar métricas como **FID** para evaluar calidad de las imágenes.  
- Usar arquitecturas más modernas (StyleGAN, WGAN-GP).  



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

