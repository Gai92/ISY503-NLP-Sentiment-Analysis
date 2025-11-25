import os
from typing import List, Tuple

# Domains used from the Multi-Domain Sentiment Dataset
DOMAINS = ["books", "dvd", "electronics", "kitchen_&_housewares"]


def get_data_directory() -> str:
    """
    Resolve the absolute path to the data directory.

    The function computes the project root based on the location of this file
    and then appends the 'data' folder. This makes the code independent of
    the current working directory.
    """
    # .../ISY503-NLP-Sentiment-Analysis/src/explore_data.py
    current_file = os.path.abspath(__file__)
    # .../ISY503-NLP-Sentiment-Analysis/src
    src_dir = os.path.dirname(current_file)
    # .../ISY503-NLP-Sentiment-Analysis
    project_root = os.path.dirname(src_dir)
    # .../ISY503-NLP-Sentiment-Analysis/data
    data_dir = os.path.join(project_root, "data")
    return data_dir


def parse_reviews(file_path: str) -> List[str]:
    """
    Parse a .review file from the Multi-Domain Sentiment Dataset.

    Each file contains multiple <review>...</review> blocks.
    This function splits the file into individual raw review strings.
    """
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


def load_all_domains(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load labelled reviews from all configured domains.

    For each domain, positive and negative .review files are parsed and
    appended to the corresponding corpus lists. The function returns the
    aggregated positive and negative review collections.
    """
    all_positive: List[str] = []
    all_negative: List[str] = []

    print("1. Loading labelled data from raw files...")

    for domain in DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        pos_file = os.path.join(domain_path, "positive.review")
        neg_file = os.path.join(domain_path, "negative.review")

        if not os.path.isfile(pos_file) or not os.path.isfile(neg_file):
            print(f"   ⚠ Domain '{domain}' was skipped: review files not found.")
            continue

        pos_reviews = parse_reviews(pos_file)
        neg_reviews = parse_reviews(neg_file)

        all_positive.extend(pos_reviews)
        all_negative.extend(neg_reviews)

        print(
            f"   Domain '{domain}': "
            f"{len(pos_reviews)} positive, {len(neg_reviews)} negative"
        )

    return all_positive, all_negative


def explore_dataset() -> None:
    """
    Perform initial data exploration.

    This step aggregates basic corpus statistics across all domains and
    prints raw samples from the positive and negative classes to support
    manual inspection of the dataset.
    """
    data_dir = get_data_directory()

    positive_reviews, negative_reviews = load_all_domains(data_dir)

    total_pos = len(positive_reviews)
    total_neg = len(negative_reviews)
    total = total_pos + total_neg

    print(
        f"\nLoaded {total} labelled reviews | "
        f"Positive: {total_pos} | Negative: {total_neg}"
    )

    if positive_reviews:
        print("\n--- Positive review sample (raw, first 200 characters) ---")
        print(positive_reviews[0][:200])

    if negative_reviews:
        print("\n--- Negative review sample (raw, first 200 characters) ---")
        print(negative_reviews[0][:200])


if __name__ == "__main__":
    explore_dataset()
