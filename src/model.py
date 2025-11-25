# src/model.py

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    Conv1D,
    GlobalMaxPooling1D,
    BatchNormalization,
    Input,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def create_lstm_model(vocab_size: int, max_length: int) -> Sequential:
    """
    Create an LSTM-based model for sentiment classification.
    """
    model = Sequential(
        [
            Embedding(vocab_size, 128, input_length=max_length),
            Bidirectional(LSTM(64, dropout=0.5, return_sequences=True)),
            Bidirectional(LSTM(32, dropout=0.5)),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )

    return model


def create_cnn_model(vocab_size: int, max_length: int) -> Sequential:
    """
    Create a CNN-based model for sentiment classification.
    """
    model = Sequential(
        [
            Embedding(vocab_size, 128, input_length=max_length),
            Conv1D(128, 5, activation="relu"),
            BatchNormalization(),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )

    return model


def create_hybrid_model(vocab_size: int, max_length: int) -> Model:
    """
    Create a hybrid CNN–LSTM model for sentiment classification.
    """
    inputs = Input(shape=(max_length,))

    # Shared embedding layer
    embedding = Embedding(vocab_size, 128)(inputs)

    # CNN branch
    conv = Conv1D(64, 5, activation="relu")(embedding)
    conv = GlobalMaxPooling1D()(conv)

    # LSTM branch
    lstm = Bidirectional(LSTM(32, dropout=0.5))(embedding)

    # Concatenate features from both branches
    merged = concatenate([conv, lstm])

    dense = Dense(64, activation="relu")(merged)
    dense = Dropout(0.5)(dense)
    outputs = Dense(1, activation="sigmoid")(dense)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )

    return model
