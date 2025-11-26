#!/usr/bin/env python3
# Quick test to check feature engineering

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_labelled_reviews, TextPreprocessor, remove_outliers
from feature_engineering import TextEncoder

def test_pipeline():
    print("Testing feature engineering pipeline...")
    print("-" * 40)
    
    # load some data
    print("Loading reviews...")
    reviews, labels = load_labelled_reviews()
    
    if len(reviews) == 0:
        print("No reviews found! Check data folder")
        return
        
    print(f"Loaded {len(reviews)} reviews")
    
    print("\nPreprocessing text...")
    preprocessor = TextPreprocessor()
    
    test_size = min(100, len(reviews))
    processed = preprocessor.preprocess_reviews(reviews[:test_size])
    test_labels = labels[:test_size]
    
    print(f"Processed {len(processed)} reviews")
    
    processed, test_labels = remove_outliers(processed, test_labels, min_length=5)
    
    # test encoding
    print("\nTesting text encoding...")
    encoder = TextEncoder(max_words=5000, max_len=150)
    encoder.fit_tokenizer(processed)
    
    # prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(
        processed, test_labels
    )
    
    print("\nSuccess! Feature engineering works")
    print(f"Final shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    print("\nSample encoded sequence (first 20 tokens):")
    print(X_train[0][:20])

if __name__ == "__main__":
    test_pipeline()
