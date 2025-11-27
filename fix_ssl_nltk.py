#!/usr/bin/env python3
# Fix SSL issue and download NLTK data

import ssl
import nltk
import os

print("Fixing SSL and downloading NLTK data...")
print("-" * 40)

# Fix SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("SSL fix applied")

# Now download NLTK data
print("\nDownloading NLTK data...")

# Set download directory
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_dir = os.path.join(project_root, "nltk_data")

# Add to NLTK path
nltk.data.path.insert(0, nltk_dir)

# Download required data
required = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4', 'punkt_tab']

for item in required:
    print(f"Downloading {item}...")
    try:
        nltk.download(item, download_dir=nltk_dir)
        print(f"  [OK] {item}")
    except Exception as e:
        print(f"  [Error] {item}: {e}")

print("\n" + "-" * 40)
print("Testing...")

# Test stopwords
try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    print(f"Stopwords: OK ({len(stop_words)} words)")
except Exception as e:
    print(f"Stopwords: FAILED - {e}")

# Test tokenizer
try:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize("This is a test.")
    print(f"Tokenizer: OK")
except Exception as e:
    print(f"Tokenizer: FAILED - {e}")

# Test lemmatizer
try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    result = lemmatizer.lemmatize("running")
    print(f"Lemmatizer: OK")
except Exception as e:
    print(f"Lemmatizer: FAILED - {e}")

print("\nDone!")
