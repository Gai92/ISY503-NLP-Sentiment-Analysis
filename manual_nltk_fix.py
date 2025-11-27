#!/usr/bin/env python3
# Manual download with SSL bypass

import os
import sys

print("Manual NLTK fix...")
print("-" * 40)

# Method 1: Use system Python to download
print("Method 1: Trying with system certificates...")
print("Run these commands in terminal:\n")

commands = """
# In Python interpreter:
python3 -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Or step by step:
python3
>>> import ssl
>>> ssl._create_default_https_context = ssl._create_unverified_context
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
>>> exit()
"""

print(commands)

print("\n" + "-" * 40)
print("Method 2: Download to home directory first")
print("-" * 40)

# Try downloading to home directory
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

# Download to default location first
print("Downloading to default NLTK location...")
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
    print("Downloaded successfully!")
    
    # Now copy to project
    print("\nNow the data should be in your home directory.")
    print("NLTK will find it automatically.")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "-" * 40)
print("Testing if it works now...")

# Add project nltk_data to path
project_root = os.path.dirname(os.path.abspath(__file__))
nltk_dir = os.path.join(project_root, "nltk_data")
nltk.data.path.insert(0, nltk_dir)

# Also add home directory paths
import pathlib
home = str(pathlib.Path.home())
nltk.data.path.append(os.path.join(home, 'nltk_data'))
nltk.data.path.append('/usr/local/share/nltk_data')
nltk.data.path.append('/usr/share/nltk_data')

print("NLTK will search in these paths:")
for path in nltk.data.path[:5]:
    print(f"  - {path}")

# Test
try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    print(f"\nSUCCESS! Stopwords loaded: {len(stop_words)} words")
    
    # Update preprocessing.py to use this fix
    print("\nYour preprocessing should work now!")
    
except Exception as e:
    print(f"\nStill not working: {e}")
    print("\nLast resort: Install certificates")
    print("Run in terminal:")
    print("pip install --upgrade certifi")
    print("Or on macOS:")
    print("/Applications/Python\\ 3.11/Install\\ Certificates.command")
