import os
import re
from typing import List, Tuple

# Domains used from the Multi-Domain Sentiment Dataset
DOMAINS = ["books", "dvd", "electronics", "kitchen_&_housewares"]


def get_data_directory() -> str:
    """
    Resolve the absolute path to the data directory based on the current file.
    """
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)
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


def load_labelled_reviews() -> Tuple[List[str], List[int]]:
    """
    Load labelled reviews from all configured domains.

    Returns:
        texts  – list of raw review strings
        labels – list of integer sentiment labels (1 = positive, 0 = negative)
    """
    data_dir = get_data_directory()

    texts: List[str] = []
    labels: List[int] = []

    print("Loading labelled data from all domains...")

    for domain in DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        pos_file = os.path.join(domain_path, "positive.review")
        neg_file = os.path.join(domain_path, "negative.review")

        if not os.path.isfile(pos_file) or not os.path.isfile(neg_file):
            print(f"   ⚠ Domain '{domain}' was skipped: review files not found.")
            continue

        pos_reviews = parse_reviews(pos_file)
        neg_reviews = parse_reviews(neg_file)

        texts.extend(pos_reviews)
        labels.extend([1] * len(pos_reviews))

        texts.extend(neg_reviews)
        labels.extend([0] * len(neg_reviews))

        print(
            f"   Domain '{domain}': "
            f"{len(pos_reviews)} positive, {len(neg_reviews)} negative"
        )

    print(f"Loaded {len(texts)} labelled reviews in total.")
    return texts, labels


class TextPreprocessor:
    """
    Text cleaning utilities for raw review strings.

    The cleaning pipeline includes:
      * lowercasing
      * removal of HTML tags
      * removal of URLs and email addresses
      * removal of non-alphabetic characters
      * removal of most single-character tokens
      * normalisation of whitespace
    """

    HTML_TAG_RE = re.compile(r"<[^>]+>")
    URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
    EMAIL_RE = re.compile(r"\S+@\S+")
    NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]+")

    def clean_text(self, text: str) -> str:
        """Apply a sequence of cleaning operations to a single review."""
        # Lowercase
        text = text.lower()

        # Remove HTML tags
        text = self.HTML_TAG_RE.sub(" ", text)

        # Remove URLs and e-mail addresses
        text = self.URL_RE.sub(" ", text)
        text = self.EMAIL_RE.sub(" ", text)

        # Keep only letters and whitespace
        text = self.NON_LETTER_RE.sub(" ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Remove most single-character tokens (keep common pronouns)
        tokens = text.split()
        filtered_tokens = [t for t in tokens if len(t) > 1 or t in {"i", "a"}]

        return " ".join(filtered_tokens)

    def transform_corpus(self, texts: List[str]) -> List[str]:
        """Clean a list of raw review strings."""
        return [self.clean_text(t) for t in texts]


def remove_outliers(texts: List[str], labels: List[int],
                    min_length: int = 5) -> Tuple[List[str], List[int]]:
    """
    Remove very short or empty reviews from the corpus.

    Args:
        texts: list of cleaned review strings
        labels: corresponding sentiment labels
        min_length: minimum number of tokens required to keep a review
    """
    filtered_texts: List[str] = []
    filtered_labels: List[int] = []

    for text, label in zip(texts, labels):
        if len(text.split()) >= min_length:
            filtered_texts.append(text)
            filtered_labels.append(label)

    print(
        f"Removed {len(texts) - len(filtered_texts)} short reviews; "
        f"{len(filtered_texts)} samples remain."
    )
    return filtered_texts, filtered_labels


def main() -> None:
    """
    Entry point for the preprocessing pipeline.

    This function performs:
      1) loading raw labelled reviews,
      2) text cleaning,
      3) basic outlier removal.
    """
    # 1. Load raw data
    raw_texts, labels = load_labelled_reviews()

    # 2. Clean text
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.transform_corpus(raw_texts)
    print("Completed text normalisation.")

    # 3. Remove very short reviews
    cleaned_texts, labels = remove_outliers(cleaned_texts, labels, min_length=5)

    # At this stage Student B can continue with tokenisation and modelling.
    # Later we will add code here to persist the cleaned corpus to disk.

    # Human-readable sanity check: show a small sample of processed reviews
    print("\n2. Preprocessing...\n")

    sample_size = 4
    print(f"--- {sample_size} processed reviews & labels ---")

    for idx in range(min(sample_size, len(cleaned_texts))):
        label = labels[idx]
        text_snippet = cleaned_texts[idx][:80]  # first ~80 characters
        print(f"Label: {label} | Text: {text_snippet}")


if __name__ == "__main__":
    main()