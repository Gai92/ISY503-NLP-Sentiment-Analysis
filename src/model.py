# model.py
# Neural network models for sentiment analysis

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional, Conv1D, GlobalMaxPooling1D,
    BatchNormalization, Input, concatenate
)
from tensorflow.keras.optimizers import Adam

def create_lstm_model(vocab_size, max_length):
    """Create LSTM model"""
    
    model = Sequential()
    
    # embedding layer
    model.add(Embedding(vocab_size, 128, input_length=max_length))
    
    # LSTM layers
    model.add(Bidirectional(LSTM(64, dropout=0.5, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.5)))
    
    # dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

def create_cnn_model(vocab_size, max_length):
    """Create CNN model for text classification"""
    
    model = Sequential()
    
    # embedding
    model.add(Embedding(vocab_size, 128, input_length=max_length))
    
    # convolutional layer
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    
    # dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    # output
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

def create_hybrid_model(vocab_size, max_length):
    """Create hybrid CNN-LSTM model
    This combines both CNN and LSTM for better performance"""
    
    # input layer
    inputs = Input(shape=(max_length,))
    
    # shared embedding
    embedding = Embedding(vocab_size, 128)(inputs)
    
    # CNN branch
    conv = Conv1D(64, 5, activation='relu')(embedding)
    conv = GlobalMaxPooling1D()(conv)
    
    # LSTM branch  
    lstm = Bidirectional(LSTM(32, dropout=0.5))(embedding)
    
    # merge both branches
    merged = concatenate([conv, lstm])
    
    # dense layers
    dense = Dense(64, activation='relu')(merged)
    dense = Dropout(0.5)(dense)
    outputs = Dense(1, activation='sigmoid')(dense)
    
    # create model
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

# test if models can be created
if __name__ == "__main__":
    print("Testing model creation...")
    
    vocab_size = 10000
    max_len = 200
    
    print("\nCreating LSTM model...")
    lstm_model = create_lstm_model(vocab_size, max_len)
    print(f"LSTM model params: {lstm_model.count_params():,}")
    
    print("\nCreating CNN model...")
    cnn_model = create_cnn_model(vocab_size, max_len)
    print(f"CNN model params: {cnn_model.count_params():,}")
    
    print("\nCreating Hybrid model...")
    hybrid_model = create_hybrid_model(vocab_size, max_len)
    print(f"Hybrid model params: {hybrid_model.count_params():,}")
    
    print("\nAll models created successfully!")
