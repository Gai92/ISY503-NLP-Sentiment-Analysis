import sys
import os
import numpy as np

# add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from preprocessing import load_labelled_reviews, TextPreprocessor, remove_outliers
from feature_engineering import TextEncoder
from model import create_lstm_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

def train_lstm():
    print("Starting LSTM training...")
    
    # load data
    print("\nLoading data...")
    reviews, labels = load_labelled_reviews()
    
    if len(reviews) == 0:
        print("No data found!")
        return
    
    print(f"Loaded {len(reviews)} reviews")
    
    # preprocess
    print("\nPreprocessing...")
    preprocessor = TextPreprocessor()
    processed_reviews = preprocessor.preprocess_reviews(reviews)
    
    # remove outliers
    processed_reviews, labels = remove_outliers(processed_reviews, labels)
    print(f"After cleaning: {len(processed_reviews)} reviews")
    
    # encode text
    print("\nEncoding text...")
    encoder = TextEncoder(max_words=10000, max_len=200)
    encoder.fit_tokenizer(processed_reviews)
    
    # prepare datasets
    X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(
        processed_reviews, labels
    )
    
    # create model
    print("\nCreating LSTM model...")
    vocab_size = min(len(encoder.tokenizer.word_index) + 1, 10000)
    model = create_lstm_model(vocab_size, 200)
    
    print("\nModel summary:")
    model.summary()
    
    # callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
    
    # create models folder if not exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    checkpoint = ModelCheckpoint(
        'models/best_lstm.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=30,  # increased from 10
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    # evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # plot training history
    plot_training_history(history)
    
    print("\nTraining complete!")
    print("Model saved to models/best_lstm.h5")
    
    return model, history

def plot_training_history(history):
    # accuracy plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    print("Saved plot to training_history.png")

# backward compatibility with old name
def plot_history(history):
    # just call the new function
    plot_training_history(history)

if __name__ == "__main__":
    train_lstm()