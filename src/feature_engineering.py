import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

MAX_WORDS = 10000
MAX_LEN = 200

class TextEncoder:
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        
    def fit_tokenizer(self, texts):
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        artifacts_dir = os.path.join(project_dir, 'artifacts')
        
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        tokenizer_path = os.path.join(artifacts_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Saved tokenizer to {tokenizer_path}")
        
    def texts_to_sequences(self, texts):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted yet!")
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, 
                              padding='post', truncating='post')
        return padded
    
    def prepare_data(self, reviews, labels, test_size=0.2, val_size=0.1):
        
        X = self.texts_to_sequences(reviews)
        y = np.array(labels)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")  
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_tokenizer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    tokenizer_path = os.path.join(project_dir, 'artifacts', 'tokenizer.pickle')
    
    if not os.path.exists(tokenizer_path):
        print("No saved tokenizer found")
        return None
        
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


if __name__ == "__main__":
    print("Testing TextEncoder...")
    
    # need enough samples for stratified split
    sample_reviews = [
        "This product is really good",
        "Terrible experience, would not recommend",
        "It's okay I guess",
        "Amazing! Best purchase ever!",
        "Worst product ever, total waste",
        "Excellent quality, highly recommend",
        "Not bad but could be better",
        "Absolutely horrible, do not buy",
        "Great value for money",
        "Complete disaster, avoid",
        "Pretty decent overall",
        "Outstanding performance",
        "Disappointing quality",
        "Worth every penny",
        "Not worth the price",
        "Satisfactory product"
    ]
    sample_labels = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 10 pos, 6 neg
    
    encoder = TextEncoder(max_words=100, max_len=10)
    encoder.fit_tokenizer(sample_reviews)
    
    encoded = encoder.texts_to_sequences(sample_reviews)
    print("\nEncoded sequences (first 5):")
    print(encoded[:5])
    
    # test with enough data
    if len(sample_reviews) >= 10:
        X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(
            sample_reviews, sample_labels
        )
        
        print("\nTrain shape:", X_train.shape)
        print("Val shape:", X_val.shape)
        print("Test shape:", X_test.shape)
    else:
        print("\nNot enough data for split test")
        
    print("\nTextEncoder test completed!")
