# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
import os
import sys

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import TextPreprocessor

app = Flask(__name__)

# Load model and tokenizer
print("Loading model...")
model = tf.keras.models.load_model('models/best_lstm.h5')

print("Loading tokenizer...")
with open('artifacts/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

preprocessor = TextPreprocessor()
MAX_LENGTH = 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        text = request.json['text']
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'success': False
            })
        
        # Preprocess text
        cleaned = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize_and_lemmatize(cleaned)
        processed = ' '.join(tokens)
        
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([processed])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=MAX_LENGTH, padding='post', truncating='post'
        )
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)[0][0]
        
        # Determine sentiment with neutral zone
        if prediction > 0.65:
            sentiment = "Positive"
            confidence = float(prediction)
        elif prediction < 0.35:
            sentiment = "Negative"
            confidence = float(1 - prediction)
        else:
            sentiment = "Neutral"
            confidence = float(1 - abs(prediction - 0.5) * 4)
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
