import os
import nltk
import ssl

# ssl fix for mac
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk_data_dir = os.path.join(project_root, "nltk_data")
nltk.data.path.append(nltk_data_dir)

import pathlib
home = str(pathlib.Path.home())
nltk.data.path.append(os.path.join(home, 'nltk_data'))

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

DOMAINS = ["books", "dvd", "electronics", "kitchen_&_housewares"]


def get_data_directory():
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)
    data_dir = os.path.join(project_root, "data")
    return data_dir


def parse_reviews(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    chunks = data.split("<review>")
    reviews = []

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


def load_labelled_reviews():
    data_dir = get_data_directory()

    reviews = []
    labels = []

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


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        
        text = re.sub(r"<.*?>", "", text)  # html tags
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        # punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)
        
        text = " ".join(text.split())

        return text

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]

        return tokens

    def preprocess_reviews(self, reviews):
        processed_reviews = []

        for i, review in enumerate(reviews):
            if i % 1000 == 0:
                print(f"Processing review {i}/{len(reviews)}")

            cleaned = self.clean_text(review)
            
            tokens = self.tokenize_and_lemmatize(cleaned)
            processed_reviews.append(" ".join(tokens))

        return processed_reviews


def remove_outliers(reviews, labels, min_length=10, max_length=1000):
    filtered_reviews = []
    filtered_labels = []
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


def check_data_quality(reviews, labels):
    df = pd.DataFrame({"review": reviews, "label": labels})

    print("\nClass distribution:")
    print(df["label"].value_counts())
    
    df["word_count"] = df["review"].apply(lambda x: len(x.split()))
    print("\nReview length statistics:")
    print(df["word_count"].describe())
    
    duplicates = df.duplicated(subset=["review"]).sum()
    print(f"\nDuplicate reviews: {duplicates}")

    return df


def main():
    raw_reviews, labels = load_labelled_reviews()

    print("\nPreprocessing...\n")
    preprocessor = TextPreprocessor()
    processed_reviews = preprocessor.preprocess_reviews(raw_reviews)

    # remove outliers
    processed_reviews, labels = remove_outliers(processed_reviews, labels)

    check_data_quality(processed_reviews, labels)
    
    sample_size = 4
    print(f"\nSample of processed reviews:")
    for idx in range(min(sample_size, len(processed_reviews))):
        label = labels[idx]
        snippet = processed_reviews[idx][:80]
        print(f"Label: {label} | Text: {snippet}")


if __name__ == "__main__":
    main()