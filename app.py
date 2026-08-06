from flask import Flask, render_template, request
import joblib
import re
import os

app = Flask(__name__)

# ---------------------------------------------------
# Load trained model and feature columns at startup
# ---------------------------------------------------
MODEL_PATH = os.path.join("model", "phishing_model.pkl")
FEATURES_PATH = os.path.join("model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)


def extract_features(url):
    """Extract the same features used during training."""
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

    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    features["has_ip"] = 1 if re.search(ip_pattern, url) else 0

    features["has_https"] = 1 if url.lower().startswith("https") else 0

    suspicious_words = ["login", "verify", "secure", "bank", "update",
                         "signin", "account", "confirm", "paypal", "webscr"]
    features["suspicious_word_count"] = sum(
        word in url.lower() for word in suspicious_words
    )

    domain_part = url.split("/")[0]
    features["domain_length"] = len(domain_part)

    # Ensure correct column order (must match training)
    return [features[col] for col in feature_columns]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = None

    if request.method == "POST":
        url = request.form["url"].strip()

        # Normalize: if URL has no path (no '/'), add trailing slash
        # to match the format used in the training dataset
        if "/" not in url:
            url = url + "/"

        features = extract_features(url)

        pred = model.predict([features])[0]
        proba = model.predict_proba([features])[0]

        if pred == 1:
            prediction = "⚠️ Suspicious URL (Possible Phishing)"
            confidence = round(proba[1] * 100, 2)
        else:
            prediction = "✅ Safe URL"
            confidence = round(proba[0] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)