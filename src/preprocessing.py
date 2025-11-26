# preprocessing.py
# handles data loading and text preprocessing

import os
import nltk

# setup nltk path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk_data_dir = os.path.join(project_root, "nltk_data")
nltk.data.path.append(nltk_data_dir)

import re
from typing import List, Tuple

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Domains used from the Multi-Domain Sentiment Dataset
DOMAINS = ["books", "dvd", "electronics", "kitchen_&_housewares"]


def get_data_directory():
    """Get path to data folder"""
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)
    data_dir = os.path.join(project_root, "data")
    return data_dir


def parse_reviews(file_path):
    """Parse review file and extract individual reviews"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    chunks = data.split("<review>")
    reviews: List[str] = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if "</review>" in chunk:
            review_text = chunk.split("</review>")[0]
        else:
            review_text = chunk

        review_text = review_text.strip()
        if review_text:
            reviews.append(review_text)

    return reviews


def load_labelled_reviews() -> Tuple[List[str], List[int]]:
    """
    Load labelled reviews from all configured domains.

    Returns:
        reviews  – list of raw review strings
        labels   – list of integer sentiment labels (1 = positive, 0 = negative)
    """
    data_dir = get_data_directory()

    reviews: List[str] = []
    labels: List[int] = []

    print("Loading labelled data from all domains...")

    for domain in DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        pos_file = os.path.join(domain_path, "positive.review")
        neg_file = os.path.join(domain_path, "negative.review")

        if not os.path.isfile(pos_file) or not os.path.isfile(neg_file):
            print(f"   Warning: Domain '{domain}' was skipped: review files not found.")
            continue

        pos_reviews = parse_reviews(pos_file)
        neg_reviews = parse_reviews(neg_file)

        reviews.extend(pos_reviews)
        labels.extend([1] * len(pos_reviews))

        reviews.extend(neg_reviews)
        labels.extend([0] * len(neg_reviews))

        print(
            f"   Domain '{domain}': "
            f"{len(pos_reviews)} positive, {len(neg_reviews)} negative"
        )

    print(f"Loaded {len(reviews)} labelled reviews in total.")
    return reviews, labels


# -----------------------------------------------------------
# Step 5: Clean and Preprocess Text
# -----------------------------------------------------------

class TextPreprocessor:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Clean individual review text."""
        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove punctuation but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text."""
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return tokens

    def preprocess_reviews(self, reviews: List[str]) -> List[str]:
        """Preprocess all reviews."""
        processed_reviews: List[str] = []

        for i, review in enumerate(reviews):
            if i % 1000 == 0:
                print(f"Processing review {i}/{len(reviews)}")

            # Clean text
            cleaned = self.clean_text(review)

            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(cleaned)

            # Join tokens back to string
            processed_reviews.append(" ".join(tokens))

        return processed_reviews


# -----------------------------------------------------------
# Step 6: Handle Outliers and Data Quality
# -----------------------------------------------------------

def remove_outliers(
    reviews: List[str],
    labels: List[int],
    min_length: int = 10,
    max_length: int = 1000,
) -> Tuple[List[str], List[int]]:
    """Remove reviews that are too short or too long."""
    filtered_reviews: List[str] = []
    filtered_labels: List[int] = []
    removed_count = 0

    for review, label in zip(reviews, labels):
        word_count = len(review.split())

        if min_length <= word_count <= max_length:
            filtered_reviews.append(review)
            filtered_labels.append(label)
        else:
            removed_count += 1

    print(f"Removed {removed_count} outlier reviews")
    return filtered_reviews, filtered_labels


def check_data_quality(reviews: List[str], labels: List[int]) -> pd.DataFrame:
    """Analyze data quality and balance."""
    df = pd.DataFrame({"review": reviews, "label": labels})

    # Check class balance
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Check review lengths
    df["word_count"] = df["review"].apply(lambda x: len(x.split()))
    print("\nReview length statistics:")
    print(df["word_count"].describe())

    # Find duplicate reviews
    duplicates = df.duplicated(subset=["review"]).sum()
    print(f"\nDuplicate reviews: {duplicates}")

    return df


# -----------------------------------------------------------
# Pipeline entry point
# -----------------------------------------------------------

def main() -> None:
    """
    End-to-end preprocessing pipeline:

      1) load raw labelled reviews,
      2) clean, tokenize and lemmatize text,
      3) remove outliers,
      4) perform basic data quality checks,
      5) print a small human-readable sample.
    """
    # 1. Load raw data
    raw_reviews, labels = load_labelled_reviews()

    # 2. Clean, tokenize and lemmatize
    print("\n2. Preprocessing...\n")
    preprocessor = TextPreprocessor()
    processed_reviews = preprocessor.preprocess_reviews(raw_reviews)

    # 3. Handle outliers
    processed_reviews, labels = remove_outliers(processed_reviews, labels)

    # 4. Data quality checks
    _ = check_data_quality(processed_reviews, labels)

    # 5. Human-readable sample
    sample_size = 4
    print(f"\n--- {sample_size} processed reviews & labels ---")
    for idx in range(min(sample_size, len(processed_reviews))):
        label = labels[idx]
        snippet = processed_reviews[idx][:80]
        print(f"Label: {label} | Text: {snippet}")


if __name__ == "__main__":
    main()