#!/usr/bin/env python3
# test_pipeline.py - test if all components work together
# This is just for testing, don't commit to git

import sys
import os
import numpy as np

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_full_pipeline():
    """Test the complete pipeline from data loading to model creation"""
    
    print("=" * 60)
    print("Testing Complete Pipeline")
    print("=" * 60)
    
    # Test data loading
    print("\n1. Testing data loading...")
    try:
        from preprocessing import load_labelled_reviews
        reviews, labels = load_labelled_reviews()
        print(f"   Loaded {len(reviews)} reviews")
        print(f"   Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
        
        if len(reviews) == 0:
            print("   ERROR: No reviews found! Check data folder")
            return False
    except Exception as e:
        print(f"   ERROR in data loading: {e}")
        return False
    
    # Test preprocessing
    print("\n2. Testing preprocessing...")
    try:
        from preprocessing import TextPreprocessor, remove_outliers
        
        preprocessor = TextPreprocessor()
        
        # just process first 100 for speed
        sample_size = min(100, len(reviews))
        sample_reviews = reviews[:sample_size]
        sample_labels = labels[:sample_size]
        
        processed = preprocessor.preprocess_reviews(sample_reviews)
        print(f"   Processed {len(processed)} reviews")
        
        # show example
        if len(processed) > 0:
            original = sample_reviews[0][:100]
            cleaned = processed[0][:100]
            print(f"   Original: '{original}...'")
            print(f"   Cleaned:  '{cleaned}...'")
        
        # remove outliers
        processed, sample_labels = remove_outliers(processed, sample_labels, min_length=5, max_length=500)
        print(f"   After outlier removal: {len(processed)} reviews")
        
    except Exception as e:
        print(f"   ERROR in preprocessing: {e}")
        return False
    
    # Test feature engineering
    print("\n3. Testing feature engineering...")
    try:
        from feature_engineering import TextEncoder
        
        encoder = TextEncoder(max_words=5000, max_len=150)
        encoder.fit_tokenizer(processed)
        
        # prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(
            processed, sample_labels, test_size=0.2, val_size=0.1
        )
        
        print(f"   Data shapes:")
        print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        print(f"   X_test:  {X_test.shape}, y_test:  {y_test.shape}")
        
        # check encoding
        print(f"   Sample encoded sequence (first 10 tokens): {X_train[0][:10]}")
        
    except Exception as e:
        print(f"   ERROR in feature engineering: {e}")
        return False
    
    # Test model creation
    print("\n4. Testing model architectures...")
    try:
        from model import create_lstm_model, create_cnn_model, create_hybrid_model
        
        vocab_size = min(5000, len(encoder.tokenizer.word_index) + 1)
        max_length = encoder.max_len
        
        print(f"   Creating models with vocab_size={vocab_size}, max_length={max_length}")
        
        # test LSTM
        print("   Testing LSTM...")
        lstm_model = create_lstm_model(vocab_size, max_length)
        lstm_output = lstm_model.predict(X_train[:2], verbose=0)
        print(f"   LSTM output shape: {lstm_output.shape}")
        print(f"   LSTM parameters: {lstm_model.count_params():,}")
        
        # test CNN
        print("   Testing CNN...")
        cnn_model = create_cnn_model(vocab_size, max_length)
        cnn_output = cnn_model.predict(X_train[:2], verbose=0)
        print(f"   CNN output shape: {cnn_output.shape}")
        print(f"   CNN parameters: {cnn_model.count_params():,}")
        
        # test Hybrid
        print("   Testing Hybrid...")
        hybrid_model = create_hybrid_model(vocab_size, max_length)
        hybrid_output = hybrid_model.predict(X_train[:2], verbose=0)
        print(f"   Hybrid output shape: {hybrid_output.shape}")
        print(f"   Hybrid parameters: {hybrid_model.count_params():,}")
        
    except Exception as e:
        print(f"   ERROR in model creation: {e}")
        return False
    
    # Quick training test (just 1 epoch to see if it works)
    print("\n5. Testing if model can train...")
    try:
        print("   Training LSTM for 1 epoch (just to test)...")
        history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=1
        )
        
        # evaluate
        loss, accuracy, auc = lstm_model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test loss: {loss:.4f}")
        print(f"   Test accuracy: {accuracy:.4f}")
        print(f"   Test AUC: {auc:.4f}")
        
    except Exception as e:
        print(f"   ERROR in training: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("SUCCESS! All components work together!")
    print("=" * 60)
    
    return True

def quick_test():
    """Even quicker test with minimal data"""
    print("\nRunning quick test with minimal data...")
    print("-" * 40)
    
    try:
        # import everything
        from preprocessing import TextPreprocessor
        from feature_engineering import TextEncoder
        from model import create_lstm_model
        
        # create dummy data
        dummy_reviews = [
            "This product is amazing and works great",
            "Terrible quality, waste of money",
            "Good value for the price",
            "Completely broken, very disappointed",
            "Excellent service and fast shipping",
            "Not worth it at all",
            "Pretty good overall",
            "Horrible experience would not recommend"
        ]
        dummy_labels = [1, 0, 1, 0, 1, 0, 1, 0]
        
        # preprocess
        preprocessor = TextPreprocessor()
        processed = []
        for review in dummy_reviews:
            cleaned = preprocessor.clean_text(review)
            tokens = preprocessor.tokenize_and_lemmatize(cleaned)
            processed.append(' '.join(tokens))
        
        # encode
        encoder = TextEncoder(max_words=100, max_len=20)
        encoder.fit_tokenizer(processed)
        X = encoder.texts_to_sequences(processed)
        y = np.array(dummy_labels)
        
        # create model
        model = create_lstm_model(100, 20)
        
        # test prediction
        output = model.predict(X[:2], verbose=0)
        
        print("Quick test passed!")
        print(f"Model output: {output.flatten()}")
        
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # first try quick test
    if not quick_test():
        print("\nQuick test failed. Check your installation.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Now testing with real data...")
    print("=" * 60)
    
    # then try full pipeline
    success = test_full_pipeline()
    
    if success:
        print("\nEverything works! Your pipeline is ready.")
    else:
        print("\nSome components failed. Check the errors.")
        print("Check:")
        print("All dependencies are installed (pip install -r requirements.txt)")
        print("Data files are in the data/ folder")
        print("NLTk data is downloaded")
