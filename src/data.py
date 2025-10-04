import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

class CIFAR10Data:
    """Clase para manejar el dataset CIFAR-10"""

    def __init__(self):
        self.class_names = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 
                           'perro', 'rana', 'caballo', 'barco', 'camión']
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def load_data(self):
        """Cargar y preparar datos de CIFAR-10"""
        print("Cargando dataset CIFAR-10...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Normalizar imágenes
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Convertir labels a one-hot encoding
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)
        
        print(f"Datos cargados:")
        print(f"X_train: {self.x_train.shape}")
        print(f"X_test: {self.x_test.shape}")
        print(f"Clases: {self.class_names}")
        
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def get_class_names(self):
        """Obtener nombres de las clases"""
        return self.class_names
    
    def get_sample_images(self, num_samples=10):
        return self.x_train[:num_samples], self.y_train[:num_samples]

# Funciones de utilidad para datos
def preprocess_images(images):
    """Normalizar"""
    return images.astype('float32') / 255.0

def show_data_shapes(x_train, y_train, x_test, y_test):
    """Mostrar formas de los datos"""
    print(f"X_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")