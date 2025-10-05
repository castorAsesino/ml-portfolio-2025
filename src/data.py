import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import os
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

class FashionMNISTData:
    def __init__(self):
        self.class_names = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
                           'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
    
    def load_data(self):
        """Cargar y preprocesar el dataset Fashion-MNIST"""
        print("Cargando dataset Fashion-MNIST...")
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        
        # Redimensionar para añadir canal (28, 28) -> (28, 28, 1)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Normalizar los datos
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Guardar versiones originales para visualización
        self.x_test_original = x_test.copy()
        self.y_test_original = y_test.copy()
        
        # Convertir etiquetas a one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        print("Datos cargados:")
        print(f"X_train: {x_train.shape}")
        print(f"X_test: {x_test.shape}")
        print(f"Clases: {self.class_names}")
        
        return x_train, y_train, x_test, y_test
    
    def get_class_names(self):
        """Obtener los nombres de las clases"""
        return self.class_names
    
    def get_test_data_original(self):
        """Obtener datos de test originales (sin one-hot)"""
        return self.x_test_original, self.y_test_original
    
    def get_sample_images(self, num_samples=5):
        """Obtener algunas imágenes de ejemplo para visualización"""
        (x_train, y_train), _ = fashion_mnist.load_data()
        indices = np.random.choice(len(x_train), num_samples, replace=False)
        return x_train[indices], y_train[indices]
    

# CLASE PARA GAN 