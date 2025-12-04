# LSTM Training

This implements Step 9 - training the LSTM model.

## How to run

From project root:
```bash
python run_training.py
```

Or directly:
```bash
python src/train.py
```

## What it does

1. Loads the data
2. Preprocesses text
3. Encodes with tokenizer
4. Creates LSTM model
5. Trains for 10 epochs (or until early stopping)
6. Saves best model to models/best_lstm.h5
7. Plots training history

## Output

- Model file: `models/best_hybrid.h5` (recommended for best performance)
- Training plot: `training_history.png`

## Settings

- Batch size: 32
- Max epochs: 10
- Early stopping patience: 3
- Optimizer: Adam
- Max words: 10000
- Max sequence length: 200

## Test

To quickly test if training works:
```bash
python test_training.py
```
