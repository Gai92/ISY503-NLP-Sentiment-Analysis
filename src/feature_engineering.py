# feature_engineering.py
# This file handles text encoding for the sentiment analysis

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# max words to use
MAX_WORDS = 10000
MAX_LEN = 200  # max review length

class TextEncoder:
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        
    def fit_tokenizer(self, texts):
        """Fit the tokenizer on our texts"""
        # create tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # save it for later
        # get the artifacts folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        artifacts_dir = os.path.join(project_dir, 'artifacts')
        
        # make sure folder exists
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        # save tokenizer
        tokenizer_path = os.path.join(artifacts_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Saved tokenizer to {tokenizer_path}")
        
    def texts_to_sequences(self, texts):
        """Convert texts to number sequences"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted yet!")
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        # pad sequences so they're all same length
        padded = pad_sequences(sequences, maxlen=self.max_len, 
                              padding='post', truncating='post')
        return padded
    
    def prepare_data(self, reviews, labels, test_size=0.2, val_size=0.1):
        """Split data into train, validation and test sets"""
        
        # encode the reviews first
        X = self.texts_to_sequences(reviews)
        y = np.array(labels)
        
        # first split - get test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # calculate validation size from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        # second split - get train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")  
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_tokenizer():
    """Load saved tokenizer from file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    tokenizer_path = os.path.join(project_dir, 'artifacts', 'tokenizer.pickle')
    
    if not os.path.exists(tokenizer_path):
        print("No saved tokenizer found")
        return None
        
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


# test the module
if __name__ == "__main__":
    # just a simple test
    print("Testing TextEncoder...")
    
    sample_reviews = [
        "This product is really good",
        "Terrible experience, would not recommend",
        "It's okay I guess",
        "Amazing! Best purchase ever!"
    ]
    sample_labels = [1, 0, 1, 1]
    
    encoder = TextEncoder(max_words=100, max_len=10)
    encoder.fit_tokenizer(sample_reviews)
    
    # test encoding
    encoded = encoder.texts_to_sequences(sample_reviews)
    print("\nEncoded sequences:")
    print(encoded)
    
    # test data split
    X_train, X_val, X_test, y_train, y_val, y_test = encoder.prepare_data(
        sample_reviews, sample_labels
    )
    
    print("\nTrain shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)
