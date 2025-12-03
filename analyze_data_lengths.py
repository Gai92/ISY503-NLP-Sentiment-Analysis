# Data length analysis
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import load_labelled_reviews

def analyze_review_lengths():
    print("Loading data...")
    reviews, labels = load_labelled_reviews()
    
    if not reviews:
        print("No data found!")
        return
    
    # Count words in each review
    lengths = [len(review.split()) for review in reviews]
    
    # Basic stats
    print(f"\nTotal reviews: {len(reviews)}")
    print(f"Average length: {np.mean(lengths):.1f} words")
    print(f"Median length: {np.median(lengths):.1f} words")
    print(f"Min length: {min(lengths)} words")
    print(f"Max length: {max(lengths)} words")
    
    # Filter out extreme outliers for better visualization
    filtered_lengths = [l for l in lengths if l <= 500]
    print(f"Showing data up to 500 words ({len(filtered_lengths)}/{len(lengths)} reviews)")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(filtered_lengths, bins=40, color='lightblue', edgecolor='black')
    ax1.set_xlabel('Number of words')
    ax1.set_ylabel('Count')
    ax1.set_title('Histogram of Review Lengths (words)')
    ax1.set_xlim(0, 500)
    
    # Boxplot
    ax2.boxplot(filtered_lengths, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen'))
    ax2.set_xlabel('Number of words')
    ax2.set_title('Boxplot of Review Lengths (words)')
    ax2.set_xlim(0, 500)
    
    plt.tight_layout()
    plt.savefig('review_lengths_analysis.png', dpi=150)
    plt.show()
    
    print("Plot saved as 'review_lengths_analysis.png'")
    
    # Check by sentiment
    pos_lengths = [lengths[i] for i in range(len(lengths)) if labels[i] == 1]
    neg_lengths = [lengths[i] for i in range(len(lengths)) if labels[i] == 0]
    
    print(f"\nPositive reviews avg length: {np.mean(pos_lengths):.1f} words")
    print(f"Negative reviews avg length: {np.mean(neg_lengths):.1f} words")

if __name__ == "__main__":
    analyze_review_lengths()
