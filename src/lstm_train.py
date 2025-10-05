import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 5                 # como pediste: 5 epochs
    lr: float = 5e-4                # LR más bajo para generalizar
    checkpoint_path: str = "results/lstm_imdb_best.h5"
    # callbacks tunables
    es_patience: int = 3
    rlp_patience: int = 2
    min_lr: float = 1e-5


class TextTrainer:
    """
    Entrenador simple para modelos de texto (IMDB) compatible con tus utils.
    Evita tabs mezclados (todo 4 espacios).
    """

    def __init__(self, model, model_name: str = "lstm_imdb_baseline", config: TrainConfig = TrainConfig()):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.history = None
        self._training_time = 0.0

    def compile(self):
        opt = tf.keras.optimizers.Adam(learning_rate=min(self.config.lr, 5e-4))
        self.model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    def train(self, x_train, y_train, x_val, y_val):
        callbacks = [
            ModelCheckpoint(
                self.config.checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config.rlp_patience,
                min_lr=self.config.min_lr,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.es_patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        start = time.time()
        self.history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            verbose=1,
            callbacks=callbacks,
        )
        self._training_time = time.time() - start
        return self.history

    def evaluate(self, x_test, y_test) -> Tuple[float, float]:
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        return loss, acc

    def predict_proba(self, x):
        """Devuelve probabilidades (sigmoid) en [0,1], shape (N,)."""
        return self.model.predict(x, verbose=0).ravel()

    def predict_labels(self, x, threshold: float = 0.5):
        """Umbral configurable para convertir proba->clase."""
        probs = self.predict_proba(x)
        return (probs >= threshold).astype("int32")

    def tune_threshold(self, x_val, y_val):
        """Busca el mejor threshold por F1 en validación."""
        probs = self.predict_proba(x_val)
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.3, 0.7, 41):  # barrido fino 0.30..0.70
            preds = (probs >= t).astype("int32")
            f1 = f1_score(y_val, preds, average="binary")
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t, best_f1

    def f1_binary(self, y_true, y_pred) -> float:
        return f1_score(y_true, y_pred, average="binary")

    def count_params(self) -> int:
        return self.model.count_params()

    def get_training_time(self) -> float:
        return self._training_time
