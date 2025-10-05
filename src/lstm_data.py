
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

class IMDBData:
    """
    Carga IMDB y prepara splits: train/val/test.
    Incluye helpers para decodificar textos y clase names.
    """
    def __init__(self, num_words=20000, maxlen=256, random_state=42):
        self.num_words = num_words
        self.maxlen = maxlen
        self.random_state = random_state
        self.class_names = ["negativo", "positivo"]
        self._index_to_word = None

    def load_data(self):
        """
        Devuelve (x_train, y_train), (x_test, y_test) con padding/truncado 'post'.
        Úsalo solo si luego vas a hacer tu propio split de validación.
        """
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.num_words)
        x_train = pad_sequences(x_train, maxlen=self.maxlen, padding="post", truncating="post")
        x_test  = pad_sequences(x_test,  maxlen=self.maxlen, padding="post", truncating="post")
        self.y_test_original = y_test.copy()
        return (x_train, y_train), (x_test, y_test)

    def load_data_with_val(self, val_size=0.2):
        """
        Carga IMDB y crea split train/val (estratificado).
        Devuelve: (x_tr, y_tr), (x_val, y_val), (x_test, y_test)
        """
        (x_train, y_train), (x_test, y_test) = self.load_data()
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=self.random_state
        )
        return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)

    def get_class_names(self):
        return self.class_names

    def vocab_size(self):
        return self.num_words

    # ---------- Helpers para decodificar (útil en “buenos/malos”) ----------
    def _build_index_to_word(self):
        if self._index_to_word is not None:
            return
        word_index = imdb.get_word_index()
        # En Keras IMDB: 0=<PAD>, 1=<START>, 2=<UNK>, 3=<UNUSED>
        self._index_to_word = {idx + 3: word for word, idx in word_index.items()}
        self._index_to_word[0] = "<PAD>"
        self._index_to_word[1] = "<START>"
        self._index_to_word[2] = "<UNK>"
        self._index_to_word[3] = "<UNUSED>"

    def decode_review(self, seq):
        """
        Recibe una secuencia de IDs (ya padded) y devuelve texto aproximado.
        """
        self._build_index_to_word()
        return " ".join(self._index_to_word.get(i, "<UNK>") for i in seq if i != 0)