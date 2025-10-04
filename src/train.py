# train.py - Funciones de entrenamiento y evaluación
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

class CIFAR10Trainer:
    """Clase para entrenar y evaluar modelos en CIFAR-10"""
    
    def __init__(self, model, model_name="cifar10_cnn"):
        self.model = model
        self.model_name = model_name
        self.history = None
        
    def setup_callbacks(self, checkpoint_dir='../results/checkpoints', patience=10):
        """Configurar callbacks para entrenamiento"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate/plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'../results/{self.model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=64):
        """Entrenar el modelo"""
        
        callbacks = self.setup_callbacks()
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """Evaluar el modelo en test set"""
        
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        print(f"Precisión en test: {test_accuracy:.4f}")
        print(f"Pérdida en test: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def save_model(self, filepath='../results/cifar10_cnn_model.h5'):
        """Guardar modelo entrenado"""
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def plot_training_history(self, save_path='../results/training_history.png'):
        """Graficar historial de entrenamiento"""
        if self.history is None:
            print("No hay historial de entrenamiento para graficar")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Gráfica de accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Precisión del Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        ax1.grid(True)
        
        # Gráfica de loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Pérdida del Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Funciones de evaluación
def evaluate_predictions(model, x_test, y_test, class_names):
    """Evaluar predicciones y generar reportes"""
    
    # Hacer predicciones
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Reporte de clasificación
    print("\n Reporte de Clasificación:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - CIFAR-10 CNN')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred_classes, y_true_classes

def plot_predictions(x_test, y_true_classes, y_pred_classes, class_names, num_samples=15):
    """Mostrar ejemplos de predicciones"""
    # Encontrar índices correctos e incorrectos
    correct_indices = np.where(y_pred_classes == y_true_classes)[0]
    incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    # Mostrar predicciones correctas
    print("Mostrando predicciones correctas...")
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(correct_indices[:8]):
        plt.subplot(2, 4, i+1)
        plt.imshow(x_test[idx])
        pred_class = class_names[y_pred_classes[idx]]
        true_class = class_names[y_true_classes[idx]]
        plt.title(f'Real: {true_class}\nPred: {pred_class}', color='green')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../results/correct_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Mostrar predicciones incorrectas
    print("Mostrando errores de predicción...")
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(incorrect_indices[:8]):
        plt.subplot(2, 4, i+1)
        plt.imshow(x_test[idx])
        pred_class = class_names[y_pred_classes[idx]]
        true_class = class_names[y_true_classes[idx]]
        plt.title(f'Real: {true_class}\nPred: {pred_class}', color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../results/incorrect_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

