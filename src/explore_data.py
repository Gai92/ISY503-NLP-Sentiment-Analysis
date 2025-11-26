import os

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


def load_all_domains(data_dir):
    all_positive = []
    all_negative = []

    print("Loading data from files...")

    for domain in DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        pos_file = os.path.join(domain_path, "positive.review")
        neg_file = os.path.join(domain_path, "negative.review")

        if not os.path.isfile(pos_file) or not os.path.isfile(neg_file):
            print(f"   Domain '{domain}' skipped - files not found")
            continue

        pos_reviews = parse_reviews(pos_file)
        neg_reviews = parse_reviews(neg_file)

        all_positive.extend(pos_reviews)
        all_negative.extend(neg_reviews)

        print(f"   {domain}: {len(pos_reviews)} pos, {len(neg_reviews)} neg")

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
