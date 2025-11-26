#!/usr/bin/env python3
# explore_data.py - look at the dataset

import os
import sys

# add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def explore_dataset():
    """Check what data we have"""
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    print("Exploring dataset...")
    print("=" * 50)
    
    # check each domain
    domains = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
    
    total_pos = 0
    total_neg = 0
    
    for domain in domains:
        domain_path = os.path.join(data_dir, domain)
        
        if not os.path.exists(domain_path):
            print(f"\n{domain}: NOT FOUND")
            continue
            
        pos_file = os.path.join(domain_path, 'positive.review')
        neg_file = os.path.join(domain_path, 'negative.review')
        
        # check if files exist
        has_pos = os.path.exists(pos_file)
        has_neg = os.path.exists(neg_file)
        
        print(f"\n{domain}:")
        
        if has_pos:
            # count reviews (rough estimate by counting <review> tags)
            with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                count = content.count('<review>')
                print(f"  Positive reviews: ~{count}")
                total_pos += count
        else:
            print(f"  Positive reviews: FILE NOT FOUND")
            
        if has_neg:
            with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                count = content.count('<review>')
                print(f"  Negative reviews: ~{count}")
                total_neg += count
        else:
            print(f"  Negative reviews: FILE NOT FOUND")
    
    print("\n" + "=" * 50)
    print(f"Total positive reviews: ~{total_pos}")
    print(f"Total negative reviews: ~{total_neg}")
    print(f"Total reviews: ~{total_pos + total_neg}")
    
    # show sample review if available
    print("\n" + "=" * 50)
    print("Sample review from books/positive.review:")
    print("-" * 50)
    
    sample_file = os.path.join(data_dir, 'books', 'positive.review')
    if os.path.exists(sample_file):
        with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # find first review
            start = content.find('<review>')
            if start != -1:
                end = content.find('</review>', start)
                if end != -1:
                    review = content[start+8:end].strip()
                    # show first 500 chars
                    if len(review) > 500:
                        print(review[:500] + "...")
                    else:
                        print(review)
    else:
        print("Sample file not found")

if __name__ == "__main__":
    explore_dataset()
