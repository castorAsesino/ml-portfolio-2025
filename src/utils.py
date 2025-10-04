import matplotlib.pyplot as plt
import numpy as np
import os
import json
import csv
def setup_plot_style():
    """Configurar estilo de gráficos"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True

def visualize_sample_images(x_data, y_data, class_names, num_images=10, title="Ejemplos CIFAR-10"):
    setup_plot_style()
    
    # Convertir one-hot back to labels si es necesario
    if len(y_data.shape) > 1:
        y_labels = np.argmax(y_data, axis=1)
    else:
        y_labels = y_data
    
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_data[i])
        plt.title(f'{class_names[y_labels[i]]}')
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('../results/sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_training_results(history, test_accuracy, model_name, filepath='../results/training_results.json'):
    results = {
        'model_name': model_name,
        'final_training_accuracy': float(history.history['accuracy'][-1]),
        'final_validation_accuracy': float(history.history['val_accuracy'][-1]),
        'final_training_loss': float(history.history['loss'][-1]),
        'final_validation_loss': float(history.history['val_loss'][-1]),
        'test_accuracy': float(test_accuracy),
        'total_epochs_trained': len(history.history['accuracy']),
        'best_validation_accuracy': float(max(history.history['val_accuracy']))
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Resultados guardados en: {filepath}")
    return results

def print_welcome_message():
    print("=" * 60)
    print("PROYECTO CIFAR-10 - CLASIFICACIÓN CON CNN")
    print("Silvia Sandoval - Portfolio Machine Learning")
    print("=" * 60)

def print_completion_message():
    print("=" * 60)
    print("Modelo entrenado y evaluado correctamente")
    print("=" * 60)

def calculate_accuracy_by_class(y_true, y_pred, class_names):
    """Calcular precisión por clase"""
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    
    for i in range(len(y_true)):
        class_total[y_true[i]] += 1
        if y_true[i] == y_pred[i]:
            class_correct[y_true[i]] += 1
    
    class_accuracy = class_correct / class_total
    
    print("\nPrecisión por clase:")
    for i, class_name in enumerate(class_names):
        print(f"   {class_name}: {class_accuracy[i]:.3f} ({class_correct[i]:.0f}/{class_total[i]:.0f})")
    
    return class_accuracy

def create_results_directory():
    """Crear directorio de resultados si no existe"""
    os.makedirs('../results', exist_ok=True)




def save_summary_csv(results, filepath='../results/summary.csv'):
    """Guardar métricas resumidas en CSV (append si existe)"""
    header = [
        'model_name', 'final_training_accuracy', 'final_validation_accuracy',
        'final_training_loss', 'final_validation_loss', 'test_accuracy',
        'total_epochs_trained', 'best_validation_accuracy'
    ]
    
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"Métricas añadidas a {filepath}")
