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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from PIL import Image
import csv
from utils import (print_welcome_message, weights_init)
from scipy.linalg import sqrtm
from torchvision.models import inception_v3

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




# Generator Model
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)