import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import create_lstm_model, create_cnn_model, create_hybrid_model

def test_models():
    
    print("Testing models...")
    
    vocab_size = 10000
    max_length = 200
    
    # dummy data
    batch_size = 32
    dummy_input = np.random.randint(0, vocab_size, (batch_size, max_length))
    dummy_labels = np.random.randint(0, 2, (batch_size,))
    
    print(f"Test data shape: {dummy_input.shape}")
    print(f"Labels shape: {dummy_labels.shape}")
    
    # lstm
    print("\nTesting LSTM...")
    try:
        lstm = create_lstm_model(vocab_size, max_length)
        output = lstm.predict(dummy_input[:5], verbose=0)
        print(f"LSTM output: {output.shape}")
        print(f"Parameters: {lstm.count_params():,}")
    except Exception as e:
        print(f"LSTM error: {e}")
    
    # cnn
    print("\nTesting CNN...")
    try:
        cnn = create_cnn_model(vocab_size, max_length)
        output = cnn.predict(dummy_input[:5], verbose=0)
        print(f"CNN output: {output.shape}")
        print(f"Parameters: {cnn.count_params():,}")
    except Exception as e:
        print(f"CNN error: {e}")
    
    # hybrid
    print("\nTesting Hybrid...")
    try:
        hybrid = create_hybrid_model(vocab_size, max_length)
        output = hybrid.predict(dummy_input[:5], verbose=0)
        print(f"Hybrid output: {output.shape}")
        print(f"Parameters: {hybrid.count_params():,}")
    except Exception as e:
        print(f"Hybrid error: {e}")
    
    print("\nDone")

if __name__ == "__main__":
    test_models()
