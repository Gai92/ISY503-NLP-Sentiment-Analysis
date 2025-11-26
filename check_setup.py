import os

# quick check if everything is ready

print("Checking project setup...")

# check data
data_dir = "data"
domains = ["books", "dvd", "electronics", "kitchen_&_housewares"]

for domain in domains:
    pos_file = os.path.join(data_dir, domain, "positive.review")
    neg_file = os.path.join(data_dir, domain, "negative.review")
    
    if os.path.exists(pos_file) and os.path.exists(neg_file):
        print(f"{domain} - ok")
    else:
        print(f"{domain} - missing files")

# check folders
folders = ["artifacts", "models"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"created {folder}/")

print("\nTry running:")
print("python src/explore_data.py")
print("python src/preprocessing.py")
