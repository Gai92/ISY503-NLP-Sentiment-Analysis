#!/usr/bin/env python3
# Download required NLTK data

import nltk
import os

# Set the path for NLTK data
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_dir = os.path.join(project_root, "nltk_data")

# Add to NLTK path
nltk.data.path.insert(0, nltk_dir)

print("Downloading NLTK data...")
print(f"Download directory: {nltk_dir}")
print("-" * 40)

# Download required data
required_data = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'omw-1.4'  # for wordnet lemmatizer
]

for data_name in required_data:
    try:
        # Check if already exists
        nltk.data.find(f'tokenizers/{data_name}')
        print(f"[Already exists] {data_name}")
    except LookupError:
        # Download if not exists
        print(f"Downloading {data_name}...")
        try:
            nltk.download(data_name, download_dir=nltk_dir)
            print(f"[Downloaded] {data_name}")
        except Exception as e:
            print(f"[Error] Could not download {data_name}: {e}")

print("-" * 40)
print("Done!")

# Test if it works
print("\nTesting NLTK data...")
try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    print(f"Stopwords loaded: {len(stop_words)} words")
except Exception as e:
    print(f"Error loading stopwords: {e}")

try:
    from nltk.tokenize import word_tokenize
    test = word_tokenize("This is a test.")
    print(f"Tokenizer works: {test}")
except Exception as e:
    print(f"Error with tokenizer: {e}")

try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    test = lemmatizer.lemmatize("running")
    print(f"Lemmatizer works: 'running' -> '{test}'")
except Exception as e:
    print(f"Error with lemmatizer: {e}")

print("\nNLTK setup complete!")
