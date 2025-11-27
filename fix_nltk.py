#!/usr/bin/env python3
# Alternative way to download NLTK data

import nltk
import os

print("Fixing NLTK data...")
print("-" * 40)

# Download to default location
print("Method 1: Downloading to default NLTK location...")
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    print("Downloaded to default location")
except Exception as e:
    print(f"Error: {e}")

# Method 2: Download to project folder
print("\nMethod 2: Downloading to project folder...")
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_dir = os.path.join(project_root, "nltk_data")

# Create directories if they don't exist
os.makedirs(nltk_dir, exist_ok=True)
os.makedirs(os.path.join(nltk_dir, "corpora"), exist_ok=True)
os.makedirs(os.path.join(nltk_dir, "tokenizers"), exist_ok=True)

# Add to path
nltk.data.path.insert(0, nltk_dir)

try:
    nltk.download('stopwords', download_dir=nltk_dir)
    nltk.download('punkt', download_dir=nltk_dir)
    nltk.download('wordnet', download_dir=nltk_dir)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_dir)
    nltk.download('omw-1.4', download_dir=nltk_dir)
    print(f"Downloaded to {nltk_dir}")
except Exception as e:
    print(f"Error: {e}")

# Test
print("\nTesting...")
print("NLTK data paths:")
for path in nltk.data.path:
    print(f"  - {path}")

try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    print(f"\nSuccess! Loaded {len(stop_words)} stopwords")
    print(f"Sample stopwords: {stop_words[:10]}")
except Exception as e:
    print(f"\nStill not working: {e}")
    print("\nTry running this in Python interpreter:")
    print(">>> import nltk")
    print(">>> nltk.download('stopwords')")
    print(">>> nltk.download('punkt')")
    print(">>> nltk.download('wordnet')")
