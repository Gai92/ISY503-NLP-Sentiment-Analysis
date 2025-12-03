# ethical_analysis.py
import os
import sys
import pandas as pd
import numpy as np

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_labelled_reviews

def analyze_bias():
    """Analyze potential biases in the dataset"""
    print("=" * 60)
    print("ETHICAL BIAS ANALYSIS")
    print("=" * 60)
    
    # Load original data
    reviews, labels = load_labelled_reviews()
    
    print(f"\nTotal reviews: {len(reviews)}")
    print(f"Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"Negative: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    # Check for demographic bias
    print("\n1. GENDER BIAS ANALYSIS")
    print("-" * 40)
    
    # Look for gender-specific terms
    gender_terms = {
        'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'male', 'gentleman'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'female', 'lady']
    }
    
    # Count occurrences
    gender_counts = {'male': 0, 'female': 0}
    gender_sentiment = {'male_pos': 0, 'male_neg': 0, 'female_pos': 0, 'female_neg': 0}
    
    for review, label in zip(reviews, labels):
        review_lower = review.lower()
        words = review_lower.split()
        
        for gender, terms in gender_terms.items():
            if any(term in words for term in terms):
                gender_counts[gender] += 1
                if label == 1:
                    gender_sentiment[f'{gender}_pos'] += 1
                else:
                    gender_sentiment[f'{gender}_neg'] += 1
    
    print(f"Reviews mentioning male terms: {gender_counts['male']}")
    print(f"  Positive: {gender_sentiment['male_pos']}")
    print(f"  Negative: {gender_sentiment['male_neg']}")
    
    print(f"\nReviews mentioning female terms: {gender_counts['female']}")
    print(f"  Positive: {gender_sentiment['female_pos']}")
    print(f"  Negative: {gender_sentiment['female_neg']}")
    
    # Check for product category bias
    print("\n2. PRODUCT CATEGORY BIAS")
    print("-" * 40)
    
    categories = {
        'electronics': ['phone', 'computer', 'laptop', 'tablet', 'camera', 'tv', 'electronic'],
        'books': ['book', 'read', 'author', 'story', 'novel', 'chapter', 'page'],
        'kitchen': ['kitchen', 'cook', 'food', 'knife', 'pan', 'pot', 'blender'],
        'movies': ['movie', 'film', 'dvd', 'actor', 'director', 'scene', 'plot']
    }
    
    category_sentiment = {}
    
    for category, keywords in categories.items():
        pos_count = 0
        neg_count = 0
        total_count = 0
        
        for review, label in zip(reviews, labels):
            review_lower = review.lower()
            if any(keyword in review_lower for keyword in keywords):
                total_count += 1
                if label == 1:
                    pos_count += 1
                else:
                    neg_count += 1
        
        if total_count > 0:
            category_sentiment[category] = {
                'total': total_count,
                'positive': pos_count,
                'negative': neg_count,
                'pos_rate': pos_count/total_count * 100
            }
    
    for cat, stats in category_sentiment.items():
        print(f"\n{cat.capitalize()}:")
        print(f"  Total mentions: {stats['total']}")
        print(f"  Positive: {stats['positive']} ({stats['pos_rate']:.1f}%)")
        print(f"  Negative: {stats['negative']} ({100-stats['pos_rate']:.1f}%)")
    
    # Check review length bias
    print("\n3. REVIEW LENGTH BIAS")
    print("-" * 40)
    
    review_lengths = [len(review.split()) for review in reviews]
    pos_lengths = [len(reviews[i].split()) for i in range(len(reviews)) if labels[i] == 1]
    neg_lengths = [len(reviews[i].split()) for i in range(len(reviews)) if labels[i] == 0]
    
    print(f"Average review length: {np.mean(review_lengths):.1f} words")
    print(f"Positive reviews avg: {np.mean(pos_lengths):.1f} words")
    print(f"Negative reviews avg: {np.mean(neg_lengths):.1f} words")
    
    if np.mean(pos_lengths) > np.mean(neg_lengths) * 1.2:
        print("⚠️  Positive reviews are significantly longer")
    elif np.mean(neg_lengths) > np.mean(pos_lengths) * 1.2:
        print("⚠️  Negative reviews are significantly longer")
    else:
        print("✓ Review lengths are relatively balanced")

def create_bias_mitigation_report():
    """Create report on bias mitigation strategies"""
    report = """
# Ethical Considerations and Bias Mitigation Report

## 1. Dataset Bias
**Issue**: Amazon reviews may over-represent certain demographics (tech-savvy users, prime members)
**Mitigation**: 
- Implemented balanced sampling across product categories
- Used stratified train/test split to maintain class balance
- Acknowledged limitations in model documentation

## 2. Sentiment Polarity
**Issue**: Binary classification oversimplifies human emotions
**Mitigation**: 
- Added confidence scores to indicate uncertainty
- Provided probability scores instead of just binary output
- Acknowledged neutral sentiments may be misclassified

## 3. Context Sensitivity
**Issue**: Sarcasm and cultural context can mislead the model
**Mitigation**: 
- Included diverse training examples
- Clear documentation of limitations
- Recommended human review for critical decisions

## 4. Fairness
**Issue**: Model may perform differently across product types
**Mitigation**: 
- Evaluated performance across categories separately
- Included data from multiple product domains
- Documented performance variations

## 5. Transparency
**Issue**: Users need to understand model limitations
**Mitigation**: 
- Clear confidence indicators in UI
- Documentation of training data sources
- Open about model architecture and limitations

## 6. Privacy
**Issue**: Reviews may contain personal information
**Mitigation**: 
- No storage of user inputs in web application
- Processing done without logging personal data
- Clear privacy notice for users

## 7. Misuse Prevention
**Issue**: Model could be used to filter or manipulate reviews
**Mitigation**: 
- Educational purpose clearly stated
- Not suitable for automated decision-making
- Encourages human oversight

## Recommendations:
1. Regular bias audits on new data
2. Collect feedback from diverse user groups
3. Consider multi-class sentiment (not just binary)
4. Implement explanation features (why sentiment was predicted)
5. Regular retraining with updated, diverse data

## Ethical Statement:
This model is designed for educational purposes and should not be used for:
- Automated content moderation without human review
- Making decisions that affect individuals without transparency
- Filtering or suppressing legitimate customer feedback
"""
    
    with open('ethical_considerations.md', 'w') as f:
        f.write(report)
    
    print("\nEthical considerations report saved to ethical_considerations.md")

if __name__ == "__main__":
    # Run bias analysis
    analyze_bias()
    
    print("\n" + "=" * 60)
    
    # Create mitigation report
    create_bias_mitigation_report()
    
    print("\nEthical analysis complete!")
