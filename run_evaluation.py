#!/usr/bin/env python3
# run evaluation only

import sys
import os
import tensorflow as tf

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_labelled_reviews, TextPreprocessor, remove_outliers
from feature_engineering import TextEncoder
from evaluation import evaluate_model

def run_evaluation():
    print("=" * 50)
    print("Model Evaluation Script")
    print("=" * 50)
    
    # Load and preprocess data
    print("\nLoading data...")
    reviews, labels = load_labelled_reviews()
    
    preprocessor = TextPreprocessor()
    processed_reviews = preprocessor.preprocess_reviews(reviews)
    processed_reviews, labels = remove_outliers(processed_reviews, labels)
    
    # Encode text
    encoder = TextEncoder(max_words=10000, max_len=200)
    encoder.fit_tokenizer(processed_reviews)
    X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(
        processed_reviews, labels
    )
    
    # Load trained model
    model_path = 'models/best_hybrid.h5'  # or best_lstm.h5, best_cnn.h5
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train a model first using run_training.py")
        return
    
    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Run evaluation - this will generate confusion_matrix.png
    print("\nRunning evaluation...")
    evaluate_model(model, X_test, y_test, encoder)
    
    print("\nEvaluation complete!")
    print("Generated files:")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"\nError during evaluation: {e}")
