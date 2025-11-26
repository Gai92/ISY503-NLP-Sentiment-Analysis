#!/usr/bin/env python3
"""
Setup script to check and prepare the project environment.
Run this before training to ensure everything is properly configured.
"""

import os
import sys
import subprocess
import pathlib

def check_python_version():
    """Check if Python version is 3.7 or higher"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required!")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'tensorflow',
        'numpy',
        'pandas',
        'nltk',
        'scikit-learn',
        'flask',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(package)
    
    return missing_packages


def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    
    # Set NLTK data path
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
    NLTK_DIR = PROJECT_ROOT / "nltk_data"
    nltk.data.path.insert(0, str(NLTK_DIR))
    
    print("\nDownloading NLTK data...")
    
    required_data = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'punkt_tab'
    ]
    
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
            print(f"✅ NLTK {data_name} already downloaded")
        except LookupError:
            try:
                nltk.download(data_name, download_dir=str(NLTK_DIR))
                print(f"✅ Downloaded NLTK {data_name}")
            except:
                print(f"⚠️  Could not download NLTK {data_name}")


def check_data_files():
    """Check if data files are present"""
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
    data_dir = PROJECT_ROOT / "data"
    
    print("\nChecking data files...")
    
    domains = ["books", "dvd", "electronics", "kitchen_&_housewares"]
    all_present = True
    
    for domain in domains:
        domain_path = data_dir / domain
        pos_file = domain_path / "positive.review"
        neg_file = domain_path / "negative.review"
        
        if pos_file.exists() and neg_file.exists():
            print(f"✅ {domain} data files present")
        else:
            print(f"❌ {domain} data files missing")
            all_present = False
    
    return all_present


def create_directories():
    """Create necessary project directories"""
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
    
    directories = [
        "models",
        "artifacts",
        "plots",
        "web/templates"
    ]
    
    print("\nCreating project directories...")
    
    for dir_name in directories:
        dir_path = PROJECT_ROOT / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory '{dir_name}' ready")


def main():
    print("=" * 60)
    print("ISY503 NLP Sentiment Analysis - Project Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        
        response = input("\nDo you want to install missing packages now? (y/n): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("\n✅ Dependencies installed!")
    else:
        print("\n✅ All dependencies are installed!")
    
    # Download NLTK data
    try:
        download_nltk_data()
    except Exception as e:
        print(f"⚠️  Error downloading NLTK data: {e}")
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Some data files are missing!")
        print("Please download the Multi-Domain Sentiment Dataset from:")
        print("http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html")
        print("And extract the review files to the appropriate directories.")
    else:
        print("\n✅ All data files are present!")
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Run 'python run_training.py' to train the model")
    print("2. Or run 'python src/main.py' for the complete pipeline")
    print("3. After training, run 'python src/app.py' for the web interface")
    print("=" * 60)


if __name__ == "__main__":
    main()
