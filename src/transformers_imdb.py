# ejercicio_3_transformers_imdb.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import csv
from datasets import load_dataset

# Configuración
os.makedirs('../results/3_transformers_imdb', exist_ok=True)

class TextClassificationTransformer:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, max_length=256):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )

class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

def load_imdb_data(transformer, sample_size=5000):
    """Carga y prepara el dataset IMDB"""
    print("Cargando dataset IMDB...")
    
    # Cargar dataset
    dataset = load_dataset('imdb')
    
    # Tomar subset para mayor velocidad
    train_dataset = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
    test_dataset = dataset['test'].select(range(min(sample_size//2, len(dataset['test']))))
    
    # Tokenizar
    print("Tokenizando datos...")
    train_encodings = transformer.tokenize_function(train_dataset)
    test_encodings = transformer.tokenize_function(test_dataset)
    
    # Crear datasets
    train_dataset = IMDBDataset(train_encodings, train_dataset['label'])
    test_dataset = IMDBDataset(test_encodings, test_dataset['label'])
    
    return train_dataset, test_dataset

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    """Entrena el modelo transformer"""
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Iniciando entrenamiento...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        total_train_loss = 0
        train_preds = []
        train_true = []
        
        progress_bar = tqdm(train_loader, desc=f'Época {epoch+1}/{epochs} [Train]')
        for batch in progress_bar:
            optimizer.zero_grad()
            
            inputs = {key: val.to(model.device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(model.device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Predicciones
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Época {epoch+1}/{epochs} [Val]'):
                inputs = {key: val.to(model.device) for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to(model.device)
                
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f'Época {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, Tiempo: {epoch_time:.2f}s')
    
    total_time = time.time() - start_time
    return train_losses, val_losses, val_accuracies, total_time

def evaluate_model(model, test_loader):
    """Evalúa el modelo en el test set"""
    model.eval()
    test_preds = []
    test_true = []
    test_probs = []
    test_texts = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluando'):
            inputs = {key: val.to(model.device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(model.device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds, average='weighted')
    cm = confusion_matrix(test_true, test_preds)
    
    return accuracy, f1, cm, test_preds, test_true, test_probs

def plot_training_curves(train_losses, val_losses, val_accuracies, save_path):
    """Grafica las curvas de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Pérdidas
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Curvas de Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precisión
    ax2.plot(val_accuracies, label='Val Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión')
    ax2.set_title('Precisión de Validación')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, save_path):
    """Grafica la matriz de confusión"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Clasificación de Sentimiento')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_predictions(test_texts, test_true, test_preds, test_probs, save_path, num_examples=10):
    """Analiza ejemplos buenos y malos de predicciones"""
    results = []
    
    for i, (true, pred, prob) in enumerate(zip(test_true, test_preds, test_probs)):
        confidence = max(prob)
        correct = (true == pred)
        results.append({
            'text': test_texts[i] if i < len(test_texts) else f"Texto_{i}",
            'true_label': 'Positivo' if true == 1 else 'Negativo',
            'pred_label': 'Positivo' if pred == 1 else 'Negativo',
            'confidence': confidence,
            'correct': correct
        })
    
    # Separar correctos e incorrectos
    correct_predictions = [r for r in results if r['correct']]
    incorrect_predictions = [r for r in results if not r['correct']]
    
    # Mostrar ejemplos
    print("\n" + "="*50)
    print("ANÁLISIS DE PREDICCIONES")
    print("="*50)
    
    print(f"\n✅ PREDICCIONES CORRECTAS (primeros {num_examples} ejemplos):")
    for i, example in enumerate(correct_predictions[:num_examples]):
        print(f"\nEjemplo {i+1}:")
        print(f"  Texto: {example['text'][:100]}...")
        print(f"  Etiqueta Real: {example['true_label']}")
        print(f"  Predicción: {example['pred_label']} (confianza: {example['confidence']:.3f})")
    
    print(f"\n❌ PREDICCIONES INCORRECTAS (primeros {num_examples} ejemplos):")
    for i, example in enumerate(incorrect_predictions[:num_examples]):
        print(f"\nEjemplo {i+1}:")
        print(f"  Texto: {example['text'][:100]}...")
        print(f"  Etiqueta Real: {example['true_label']}")
        print(f"  Predicción: {example['pred_label']} (confianza: {example['confidence']:.3f})")
    
    # Guardar análisis en CSV
    df_analysis = pd.DataFrame(results)
    df_analysis.to_csv(save_path, index=False)
    print(f"\n📊 Análisis guardado en: {save_path}")
    
    return correct_predictions, incorrect_predictions

def save_to_summary(model_name, notebook_name, accuracy, f1, test_loss, epochs, 
                   model_params, training_time, save_path="../results/summary.csv"):
    """Guarda resultados en el archivo summary.csv"""
    
    header = ["model_name", "notebook_name", "test_accuracy", "f1", "test_loss", 
              "epochs", "model_params", "training_time"]
    
    row = [
        model_name,
        notebook_name,
        accuracy,
        f1,
        test_loss,
        epochs,
        model_params,
        training_time
    ]
    
    write_header = not os.path.exists(save_path) or os.path.getsize(save_path) == 0
    
    with open(save_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    print("="*70)
    print("TRANSFORMERS PARA CLASIFICACIÓN DE TEXTO - IMDB")
    print("="*70)
    
    # Configuración
    MODEL_NAME = "distilbert-base-uncased"  # Modelo ligero para mayor velocidad
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    SAMPLE_SIZE = 2000  # Reducir para mayor velocidad
    
    # Inicializar transformer
    print("Inicializando modelo transformer...")
    transformer = TextClassificationTransformer(model_name=MODEL_NAME)
    
    # Cargar datos
    train_dataset, test_dataset = load_imdb_data(transformer, sample_size=SAMPLE_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset de entrenamiento: {len(train_dataset)} ejemplos")
    print(f"Dataset de test: {len(test_dataset)} ejemplos")
    
    # Entrenar modelo
    train_losses, val_losses, val_accuracies, training_time = train_model(
        transformer.model, train_loader, test_loader,  # Usamos test como val para simplicidad
        epochs=EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # Evaluar modelo
    accuracy, f1, cm, test_preds, test_true, test_probs = evaluate_model(
        transformer.model, test_loader
    )
    
    # Obtener algunos textos de ejemplo (simulados para el análisis)
    test_texts = [f"Texto de ejemplo {i}" for i in range(len(test_true))]
    
    # Visualizaciones
    print("\nGenerando visualizaciones...")
    
    # Curvas de entrenamiento
    plot_training_curves(
        train_losses, val_losses, val_accuracies,
        '../results/3_transformers_imdb/training_curves.png'
    )
    
    # Matriz de confusión
    plot_confusion_matrix(cm, '../results/3_transformers_imdb/confusion_matrix.png')
    
    # Análisis de predicciones
    correct, incorrect = analyze_predictions(
        test_texts, test_true, test_preds, test_probs,
        '../results/3_transformers_imdb/predictions_analysis.csv'
    )
    
    # Guardar modelo
    transformer.model.save_pretrained('../results/3_transformers_imdb/final_model')
    transformer.tokenizer.save_pretrained('../results/3_transformers_imdb/final_model')
    
    # Guardar en summary
    model_params = sum(p.numel() for p in transformer.model.parameters())
    test_loss = val_losses[-1] if val_losses else float("nan")
    
    save_to_summary(
        model_name=f"distilbert_imdb",
        notebook_name="3_transformers_imdb",
        accuracy=accuracy,
        f1=f1,
        test_loss=test_loss,
        epochs=EPOCHS,
        model_params=model_params,
        training_time=training_time
    )
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL - TRANSFORMERS IMDB")
    print("="*70)
    print(f"✅ Precisión en test: {accuracy:.4f}")
    print(f"✅ F1-score: {f1:.4f}")
    print(f"✅ Pérdida final: {test_loss:.4f}")
    print(f"✅ Tiempo de entrenamiento: {training_time:.2f}s")
    print(f"✅ Parámetros del modelo: {model_params:,}")
    print(f"✅ Épocas: {EPOCHS}")
    print(f"\n📊 Archivos guardados:")
    print(f"   • Modelo: ../results/3_transformers_imdb/final_model/")
    print(f"   • Gráficas: training_curves.png, confusion_matrix.png")
    print(f"   • Análisis: predictions_analysis.csv")
    print(f"   • Resumen: summary.csv")
    print(f"\n📈 Métricas por época:")
    for i, (acc, loss) in enumerate(zip(val_accuracies, val_losses)):
        print(f"   Época {i+1}: Acc={acc:.4f}, Loss={loss:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()