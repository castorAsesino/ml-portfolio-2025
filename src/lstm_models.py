from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras import regularizers

def create_lstm_text_model(vocab_size: int,
                           maxlen: int = 200,
                           embed_dim: int = 128,
                           lstm_units: int = 96,
                           bidirectional: bool = True,
                           dropout: float = 0.5,
                           recurrent_dropout: float = 0.25,
                           spatial_dropout: float = 0.2):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim,
                        input_length=maxlen, mask_zero=True,
                        embeddings_regularizer=regularizers.l2(1e-6)))
    model.add(SpatialDropout1D(spatial_dropout))

    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout)))
    else:
        model.add(LSTM(lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid',
                    kernel_regularizer=regularizers.l2(1e-6)))
    return model
