"""
train_model.py
Trains a phishing URL detection model using hand-engineered
features extracted from raw URL strings.
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# ---------------------------------------------------
# STEP 1: Load dataset
# ---------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("../phishing_dataset.csv")   # path: from notebook/ folder, go up one level

print("Original shape:", df.shape)

# Remove duplicate URLs
df = df.drop_duplicates(subset="URL").reset_index(drop=True)
print("After removing duplicates:", df.shape)

# Convert label: 'bad' -> 1 (phishing), 'good' -> 0 (safe)
df["label"] = df["Label"].apply(lambda x: 1 if x.strip().lower() == "bad" else 0)


# ---------------------------------------------------
# STEP 2: Feature Engineering
# ---------------------------------------------------
def extract_features(url):
    url = str(url)
    features = {}

    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_at"] = url.count("@")
    features["num_underscore"] = url.count("_")
    features["num_slash"] = url.count("/")
    features["num_question"] = url.count("?")
    features["num_equal"] = url.count("=")
    features["num_digits"] = sum(c.isdigit() for c in url)

    # Has IP address instead of domain name
    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    features["has_ip"] = 1 if re.search(ip_pattern, url) else 0

    # Uses HTTPS
    features["has_https"] = 1 if url.lower().startswith("https") else 0

    # Suspicious keywords often used in phishing URLs
    suspicious_words = ["login", "verify", "secure", "bank", "update",
                         "signin", "account", "confirm", "paypal", "webscr"]
    features["suspicious_word_count"] = sum(
        word in url.lower() for word in suspicious_words
    )

    # Domain length (rough estimate, before first '/')
    domain_part = url.split("/")[0]
    features["domain_length"] = len(domain_part)

    return features


print("Extracting features from URLs... (this may take a minute)")
feature_list = df["URL"].apply(extract_features)
features_df = pd.DataFrame(list(feature_list))

# Combine features with label
final_df = pd.concat([features_df, df["label"]], axis=1)
print("Feature extraction complete. Shape:", final_df.shape)
print(final_df.head())


# ---------------------------------------------------
# STEP 3: Train / Test Split
# ---------------------------------------------------
X = final_df.drop("label", axis=1)
y = final_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


# ---------------------------------------------------
# STEP 4: Train Model
# ---------------------------------------------------
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)


# ---------------------------------------------------
# STEP 5: Evaluate Model
# ---------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (useful for README/interview explanation)
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Feature Importances:")
print(importance_df)


# ---------------------------------------------------
# STEP 6: Save Model
# ---------------------------------------------------
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/phishing_model.pkl")
print("\nModel saved to model/phishing_model.pkl")

# Save feature column order (needed later in app.py)
joblib.dump(list(X.columns), "../model/feature_columns.pkl")
print("Feature columns saved to model/feature_columns.pkl")
