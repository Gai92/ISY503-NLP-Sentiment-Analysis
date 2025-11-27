# How to Test Your Setup

## 1. Quick Setup Check
First, check if everything is installed correctly:

```bash
python3 check_setup.py
```

This will verify:
- Python version
- All source files exist
- Data files are present
- Dependencies are installed

## 2. Test Individual Components

### Test Data Loading:
```bash
python3 src/explore_data.py
```

### Test Preprocessing:
```bash
python3 src/preprocessing.py
```

### Test Feature Engineering:
```bash
python3 src/feature_engineering.py
```

### Test Models:
```bash
python3 src/model.py
```

## 3. Test Full Pipeline
Run the complete pipeline test:

```bash
python3 test_pipeline.py
```

This will:
1. Load some data
2. Preprocess it
3. Encode it
4. Create models
5. Do a quick training test

## Expected Output

If everything works, you should see:
- Data loads successfully
- Preprocessing works
- Feature engineering creates encodings
- Models are created with correct shapes
- Quick training runs without errors

## Troubleshooting

### If imports fail:
```bash
pip install -r requirements.txt
```

### If NLTK data missing:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### If data files missing:
Download from: http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
Extract to the data/ folder

## What's Next?

Once all tests pass, the project is ready for:
1. Training implementation (Step 9)
2. Evaluation metrics (Phase 5)
3. Web application (Phase 6)

These are tasks for other team members or next phase of development.
