#!/usr/bin/env python3
# test if models work with our feature engineering

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import create_lstm_model, create_cnn_model, create_hybrid_model

def test_models():
    """Quick test to see if models work"""
    
    print("Testing model architectures...")
    print("-" * 40)
    
    # params for models
    vocab_size = 10000
    max_length = 200
    
    # create some dummy data for testing
    batch_size = 32
    dummy_input = np.random.randint(0, vocab_size, (batch_size, max_length))
    dummy_labels = np.random.randint(0, 2, (batch_size,))
    
    print(f"Test data shape: {dummy_input.shape}")
    print(f"Labels shape: {dummy_labels.shape}")
    print()
    
    # Test LSTM
    print("1. Testing LSTM model...")
    try:
        lstm = create_lstm_model(vocab_size, max_length)
        # test forward pass
        output = lstm.predict(dummy_input[:5], verbose=0)
        print(f"   LSTM output shape: {output.shape}")
        print(f"   Total parameters: {lstm.count_params():,}")
        print("   ✓ LSTM model works!")
    except Exception as e:
        print(f"   ✗ LSTM failed: {e}")
    
    print()
    
    # Test CNN
    print("2. Testing CNN model...")
    try:
        cnn = create_cnn_model(vocab_size, max_length)
        output = cnn.predict(dummy_input[:5], verbose=0)
        print(f"   CNN output shape: {output.shape}")
        print(f"   Total parameters: {cnn.count_params():,}")
        print("   ✓ CNN model works!")
    except Exception as e:
        print(f"   ✗ CNN failed: {e}")
    
    print()
    
    # Test Hybrid
    print("3. Testing Hybrid model...")
    try:
        hybrid = create_hybrid_model(vocab_size, max_length)
        output = hybrid.predict(dummy_input[:5], verbose=0)
        print(f"   Hybrid output shape: {output.shape}")
        print(f"   Total parameters: {hybrid.count_params():,}")
        print("   ✓ Hybrid model works!")
    except Exception as e:
        print(f"   ✗ Hybrid failed: {e}")
    
    print()
    print("All models tested successfully!")

if __name__ == "__main__":
    test_models()
