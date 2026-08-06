# AI-Based Phishing Detection System

A machine learning-based system that classifies URLs as safe or potentially malicious (phishing), built with Python, Flask, and scikit-learn.

## Problem

Phishing attacks trick users into visiting malicious URLs disguised as legitimate websites. This project builds a lightweight, ML-based classifier that flags suspicious URLs in real time based on structural URL patterns, without relying on external threat-intelligence APIs.

## Approach

1. **Dataset:** ~550,000 labeled URLs (safe / phishing), sourced and cleaned (duplicates removed, ~507K unique URLs used for training).
2. **Feature Engineering:** 13 hand-crafted features extracted directly from the raw URL string — including URL length, number of dots/hyphens/slashes, presence of an IP address instead of a domain, count of suspicious keywords (e.g. "login", "verify", "secure"), and domain length.
3. **Model:** Random Forest Classifier (scikit-learn), trained with class balancing to improve detection of the minority (phishing) class.
4. **Web App:** A Flask interface where a user submits a URL and receives a real-time prediction with a confidence score.

## Results

Evaluated on a held-out 20% test set (~101K URLs):

| Metric | Score |
|---|---|
| Accuracy | 87.58% |
| Precision | 71.36% |
| Recall | 74.97% |
| F1 Score | 73.12% |

**Why recall matters more here:** In phishing detection, failing to flag an actual phishing URL (false negative) is more costly than incorrectly flagging a safe URL (false positive). The model was tuned using class-balancing to prioritize recall, improving it from 53.9% to 74.97% — at the cost of some precision. This is a deliberate design trade-off suited to a security context.

**Top predictive features:** suspicious keyword count, number of digits, number of dots, and URL length were the strongest signals for classification.

## Known Limitations

- Feature set is based on URL structure only — no domain age, WHOIS, or SSL certificate reputation checks are used, which a production system would typically include.
- Short, "bare" domain inputs (e.g. `amazon.in` without a path) sometimes produce borderline predictions (~50–55% confidence), reflecting the training data's format (most safe-labeled URLs included a trailing path).
- This is a portfolio/learning project, not a production-ready security tool.

## Tech Stack

- Python, Flask
- scikit-learn (Random Forest)
- pandas, joblib
- HTML/CSS

## Project Structure

```
AI-Phishing-Detection-System/
├── app.py                     # Flask web app (loads trained model)
├── model/
│   ├── phishing_model.pkl     # Trained Random Forest model
│   └── feature_columns.pkl    # Feature order used during training
├── notebook/
│   └── train_model.py         # Feature engineering + model training script
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── phishing_dataset.csv
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt

# (Optional) Retrain the model from scratch:
cd notebook
python train_model.py

# Run the web app:
cd ..
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Future Improvements

- Incorporate domain reputation / WHOIS-based features
- Address borderline predictions for bare-domain inputs with additional training examples
- Deploy as a public web app (Render/Railway) for live demo access
- Add a browser extension front-end