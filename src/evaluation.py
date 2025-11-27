import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import TextPreprocessor

def evaluate_model(model, X_test, y_test, encoder=None):
    """Evaluate model performance"""
    # make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Negative', 'Positive']))
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    print("Confusion matrix saved to confusion_matrix.png")
    
    # test custom reviews if encoder provided
    if encoder is not None:
        test_custom_reviews(model, encoder)
    
    return y_pred

def test_custom_reviews(model, encoder):
    """Test model on custom review examples"""
    custom_reviews = [
        "This product is absolutely fantastic! Best purchase ever!",
        "Terrible quality. Complete waste of money. Very disappointed.",
        "It's okay, nothing special but does the job.",
        "Amazing service and great product quality. Highly recommend!",
        "Broke after one day. Do not buy this garbage.",
        "Good value for money. Satisfied with my purchase."
    ]
    
    # preprocess custom reviews
    preprocessor = TextPreprocessor()
    processed = []
    for review in custom_reviews:
        cleaned = preprocessor.clean_text(review)
        tokens = preprocessor.tokenize_and_lemmatize(cleaned)
        processed.append(' '.join(tokens))
    
    # encode and predict
    encoded = encoder.texts_to_sequences(processed)
    predictions = model.predict(encoded)
    
    print("\nCustom Review Predictions:")
    for review, pred in zip(custom_reviews, predictions):
        sentiment = "Positive" if pred > 0.5 else "Negative"
        confidence = pred[0] if pred > 0.5 else 1 - pred[0]
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})\n")

def test_on_examples(model, encoder, preprocessor):
    """Legacy function for compatibility"""
    test_reviews = [
        "This product is amazing! Best purchase ever!",
        "Terrible quality, waste of money",
        "It's okay, nothing special",
        "Absolutely love it! Highly recommend!"
    ]
    
    print("\nTesting on examples:")
    for review in test_reviews:
        # preprocess
        cleaned = preprocessor.clean_text(review)
        tokens = preprocessor.tokenize_and_lemmatize(cleaned)
        processed = ' '.join(tokens)
        
        # encode
        encoded = encoder.texts_to_sequences([processed])
        
        # predict
        pred = model.predict(encoded, verbose=0)[0][0]
        sentiment = "Positive" if pred > 0.5 else "Negative"
        confidence = pred if pred > 0.5 else 1 - pred
        
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")
        print()
