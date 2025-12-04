# Sentiment Analysis Project

## Overview
This project implements a sentiment analysis system for Amazon product reviews using deep learning techniques. It includes data preprocessing, multiple neural network architectures (LSTM, CNN, Hybrid), and a web interface for real-time predictions.

## Project Structure
```
sentiment-analysis-project/
├── data/                   # Dataset files
│   ├── books/
│   ├── dvd/
│   ├── electronics/
│   └── kitchen_&_housewares/
├── src/                   # Source code
│   ├── explore_data.py    # Data exploration
│   ├── preprocessing.py   # Text preprocessing
│   ├── feature_engineering.py  # Text encoding
│   ├── model.py           # Neural network models
│   ├── train.py           # Training functions
│   └── evaluation.py      # Model evaluation
├── models/                # Saved models
├── artifacts/             # Tokenizer and other artifacts
├── templates/             # HTML templates for web app
├── app.py                 # Flask web application
├── run_evaluation.py      # Model evaluation and visualization
├── run_training.py        # LSTM training script
├── run_cnn_training.py    # CNN training script
├── run_hybrid_training.py # Hybrid model training script
├── ethical_analysis.py   # Bias analysis
└── requirements.txt      # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd sentiment-analysis-project
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```bash
python download_nltk_data.py
```

## Dataset
Download the Amazon product reviews dataset from:
http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

Extract the files into the `data/` directory maintaining the structure:
- data/books/
- data/dvd/
- data/electronics/
- data/kitchen_&_housewares/

## Usage

### 1. Explore Data
```bash
python src/explore_data.py
```

### 2. Train Models

```bash
python run_training.py        # Train LSTM model
python run_cnn_training.py    # Train CNN model
python run_hybrid_training.py # Train Hybrid model (recommended)
```

Or use training functions directly:
```bash
python src/train.py           # Direct access to training functions
```

Each training automatically generates confusion matrix and training history plots.

### 3. Web Application
```bash
python app.py
```
Then open http://localhost:5000 in your browser.

### 4. Model Evaluation & Visualization
Generate confusion matrix and performance metrics:
```bash
python run_evaluation.py
```

This will create:
- `confusion_matrix.png` - Confusion matrix for the specified model
- Performance metrics (accuracy, precision, recall, F1-score)
- Custom review predictions

**Note**: When training models individually, each generates its own confusion matrix:
- `python run_training.py` → `lstm_confusion_matrix.png`
- `python run_cnn_training.py` → `cnn_confusion_matrix.png`  
- `python run_hybrid_training.py` → `hybrid_confusion_matrix.png`

### 5. Ethical Analysis
```bash
python ethical_analysis.py
```

## Models

### LSTM Model
- Bidirectional LSTM layers
- Dropout for regularization
- Best for capturing sequential patterns

### CNN Model
- 1D Convolutional layers
- Global max pooling
- Fast training and inference

### Hybrid Model **Recommended**
- Combines CNN and LSTM
- Captures both local and sequential features
- Best overall performance (~86% accuracy)

## Performance

| Model  | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| LSTM   | ~85%     | ~86%      | ~84%   | ~85%     | ~92%|
| CNN    | ~83%     | ~84%      | ~82%   | ~83%     | ~90%|
| Hybrid | ~86%     | ~87%      | ~85%   | ~86%     | ~93%|

*Note: Actual results may vary depending on training*

## Features

- **Data Preprocessing**: Cleaning, tokenization, lemmatization
- **Multiple Models**: LSTM, CNN, and Hybrid architectures
- **Web Interface**: Real-time sentiment prediction
- **Automated Visualization**: Training history plots, confusion matrices
- **Performance Evaluation**: Detailed metrics and custom review testing
- **Ethical Analysis**: Bias detection and mitigation strategies

## API Endpoints

### POST /predict
Predicts sentiment for given text.

Request:
```json
{
    "text": "Your review text here"
}
```

Response:
```json
{
    "sentiment": "Positive",
    "confidence": 0.92,
    "success": true
}
```

## Ethical Considerations

This project includes comprehensive bias analysis:
- Gender bias detection
- Product category bias analysis
- Review length bias examination
- Mitigation strategies documented

See `ethical_considerations.md` for detailed report.

## Limitations

1. Binary classification (positive/negative only)
2. English language only
3. Trained on Amazon reviews (may not generalize to other domains)
4. Sarcasm detection limited
5. Requires minimum text length for accurate predictions

## Future Improvements

1. Multi-class sentiment (very positive, positive, neutral, negative, very negative)
2. Aspect-based sentiment analysis
3. Multi-language support
4. Real-time model updating
5. Explainable AI features
6. Mobile application

## Generated Outputs

After training and evaluation, the following files are automatically created:

### Training Visualizations
- `lstm_training_history.png` - LSTM training curves
- `cnn_training_history.png` - CNN training curves  
- `hybrid_training_history.png` - Hybrid model training curves

### Performance Analysis
- `lstm_confusion_matrix.png` - LSTM model confusion matrix
- `cnn_confusion_matrix.png` - CNN model confusion matrix  
- `hybrid_confusion_matrix.png` - Hybrid model confusion matrix
- Model performance metrics printed to console for each model

### Saved Models
- `models/best_lstm.h5` - Best LSTM model weights
- `models/best_cnn.h5` - Best CNN model weights
- `models/best_hybrid.h5` - Best Hybrid model weights
- `artifacts/tokenizer.pickle` - Fitted tokenizer for deployment

## Testing

Run tests:
```bash
python test_pipeline.py
```

Run evaluation only (if models already trained):
```bash
python run_evaluation.py
```
