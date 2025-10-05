import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import time
from datetime import datetime
import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import time
from datetime import datetime
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf



def print_welcome_message(project_name="Fashion-MNIST CNN"):
    """Imprimir mensaje de bienvenida"""
    print("=" * 60)
    print(f"PROYECTO {project_name.upper()}")
    print("Silvia Sandoval - Portfolio Machine Learning")
    print("=" * 60)

def print_completion_message():
    """Imprimir mensaje de finalización"""
    print("=" * 60)
    print("Modelo entrenado y evaluado correctamente")
    print("=" * 60)

def create_results_directory(notebook_name):
    """Crear directorio de resultados específico para el notebook"""
    results_dir = f"../results/{notebook_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Directorio creado: {results_dir}")
    return results_dir

def save_training_results(history, test_accuracy, test_loss, f1, model_name, notebook_name, training_time, model_params):
    """Guardar resultados del entrenamiento en JSON"""
    results_dir = create_results_directory(notebook_name)
    
    results = {
        'model_name': model_name,
        'notebook_name': notebook_name,
        'final_accuracy': float(test_accuracy),
        'final_loss': float(test_loss),
        'f1_score': float(f1),
        'training_time_seconds': float(training_time),
        'model_params': int(model_params),
        'timestamp': datetime.now().isoformat(),
        'training_history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }
    
    results_path = f"{results_dir}/training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Resultados guardados en: {results_path}")
    return results

def save_summary_csv(results, notebook_name):
    """Guardar métricas en archivo CSV de resumen general"""
    csv_path = "../results/summary.csv"
    
    # Crear DataFrame con las métricas
    summary_data = {
        'notebook_name': [results['notebook_name']],
        'model_name': [results['model_name']],
        'final_accuracy': [results['final_accuracy']],
        'final_loss': [results['final_loss']],
        'f1_score': [results['f1_score']],
        'training_time_seconds': [results['training_time_seconds']],
        'model_params': [results['model_params']],
        'epochs_trained': [len(results['training_history']['accuracy'])],
        'final_training_accuracy': [results['training_history']['accuracy'][-1]],
        'final_val_accuracy': [results['training_history']['val_accuracy'][-1]],
        'final_training_loss': [results['training_history']['loss'][-1]],
        'final_val_loss': [results['training_history']['val_loss'][-1]],
        'timestamp': [results['timestamp']]
    }
    
    df = pd.DataFrame(summary_data)
    
    # Si el archivo existe, añadir los datos
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # Evitar duplicados
        mask = (existing_df['notebook_name'] == notebook_name) & (existing_df['model_name'] == results['model_name'])
        existing_df = existing_df[~mask]
        df = pd.concat([existing_df, df], ignore_index=True)
    else:
        # Crear directorio results si no existe
        os.makedirs("../results", exist_ok=True)
    
    df.to_csv(csv_path, index=False)
    print(f"Métricas añadidas al summary general: {csv_path}")

def plot_training_history(history, notebook_name, model_name):
    """Graficar la historia del entrenamiento y guardar"""
    results_dir = create_results_directory(notebook_name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de precisión
    ax1.plot(history.history['accuracy'], label='Precisión entrenamiento', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Precisión validación', linewidth=2)
    ax1.set_title(f'Precisión del Modelo - {model_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de pérdida
    ax2.plot(history.history['loss'], label='Pérdida entrenamiento', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Pérdida validación', linewidth=2)
    ax2.set_title(f'Pérdida del Modelo - {model_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    plot_path = f"{results_dir}/learning_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Curvas de aprendizaje guardadas en: {plot_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, notebook_name, model_name):
    """Graficar y guardar matriz de confusión"""
    results_dir = create_results_directory(notebook_name)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Guardar figura
    cm_path = f"{results_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Matriz de confusión guardada en: {cm_path}")
    return cm

def plot_examples(model, x_test, y_test, class_names, notebook_name, model_name, num_examples=10):
    """Mostrar ejemplos buenos y malos de predicciones"""
    results_dir = create_results_directory(notebook_name)
    
    # Predecir
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Encontrar ejemplos buenos y malos
    correct_indices = np.where(y_pred == y_true)[0]
    incorrect_indices = np.where(y_pred != y_true)[0]
    
    # Seleccionar ejemplos aleatorios
    good_examples = np.random.choice(correct_indices, min(num_examples, len(correct_indices)), replace=False)
    bad_examples = np.random.choice(incorrect_indices, min(num_examples, len(incorrect_indices)), replace=False)
    
    # Plot ejemplos buenos
    fig, axes = plt.subplots(2, num_examples, figsize=(20, 6))
    fig.suptitle(f'Ejemplos de Predicciones - {model_name}', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(good_examples):
        if i < num_examples:
            # Para Fashion-MNIST, quitar la dimensión del canal para visualizar
            img = x_test[idx].reshape(28, 28)
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Real: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}', 
                               color='green', fontsize=8)
            axes[0, i].axis('off')
    
    # Plot ejemplos malos
    for i, idx in enumerate(bad_examples):
        if i < num_examples:
            img = x_test[idx].reshape(28, 28)
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Real: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}', 
                               color='red', fontsize=8)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Guardar figura
    examples_path = f"{results_dir}/prediction_examples.png"
    plt.savefig(examples_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Ejemplos de predicciones guardados en: {examples_path}")

def plot_sample_images(data_loader, notebook_name):
    """Mostrar imágenes de ejemplo del dataset"""
    results_dir = create_results_directory(notebook_name)
    
    sample_images, sample_labels = data_loader.get_sample_images(10)
    class_names = data_loader.get_class_names()
    
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i], cmap='gray')
        plt.title(f'{class_names[sample_labels[i]]}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Ejemplos del Dataset Fashion-MNIST', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Guardar figura
    samples_path = f"{results_dir}/dataset_samples.png"
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Ejemplos del dataset guardados en: {samples_path}")

def calculate_metrics(model, x_test, y_test, class_names):
    """Calcular métricas detalladas"""
    # Predecir
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calcular F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Reporte de clasificación
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print("\n" + "="*50)
    print("MÉTRICAS DETALLADAS")
    print("="*50)
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("\nReporte por clase:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return y_true, y_pred, f1, report






#FUNCIONES PARA GAN 
# === DCGAN UTILS ===
#FUNCIONES PARA GAN 
# === DCGAN UTILS ===
