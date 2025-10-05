import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import time
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt

class FashionMNISTTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.history = None
        self.training_time = 0
        
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
        """Entrenar el modelo con callbacks y medir tiempo"""
        # Callbacks
        checkpoint_path = f"../results/{self.model_name}_best.h5"
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        callbacks = [checkpoint, reduce_lr]
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        
        # Entrenamiento
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        print(f"Tiempo de entrenamiento: {self.training_time:.2f} segundos")
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """Evaluar el modelo en el conjunto de test"""
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Precisión en test: {test_accuracy:.4f}")
        print(f"Pérdida en test: {test_loss:.4f}")
        return test_loss, test_accuracy
    
    def save_model(self, notebook_name):
        """Guardar el modelo final"""
        import os
        results_dir = f"../results/{notebook_name}"
        os.makedirs(results_dir, exist_ok=True)
        
        model_path = f"{results_dir}/{self.model_name}_model.h5"
        self.model.save(model_path)
        print(f"Modelo guardado en: {model_path}")
    
    def get_training_time(self):
        """Obtener tiempo de entrenamiento"""
        return self.training_time
    
    def get_model_params(self):
        """Obtener número de parámetros del modelo"""
        return self.model.count_params()
    