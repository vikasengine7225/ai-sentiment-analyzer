# Customer Review Sentiment Analyzer

A Streamlit application that classifies customer reviews as **positive**, **neutral**, or **negative** using TF-IDF text features and Logistic Regression.

## Features

- Analyze a single review.
- Upload a CSV file and predict sentiment for each row.
- View estimated class probabilities.
- View a demonstration evaluation report and confusion matrix.

## Requirements

Python 3.10+ is recommended. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the app

From the project root:

```bash
python -m streamlit run review-analyser-model.py
```

If your main Python file has a different name, replace `review-analyser-model.py` with that filename.

## Model files and fallback behavior

The application looks for these saved files inside `models/`:

- `models/sentiment_model.joblib`
- `models/tfidf_vectorizer.joblib`

If both files exist, the application loads them. If either is missing, the app trains a fallback model using generated synthetic review templates.

The fallback dataset is synthetic and repetitive. Evaluation on this dataset demonstrates the workflow; it does not establish real-world accuracy. For meaningful evaluation, use representative real reviews with trustworthy labels.

## Optional project structure

```text
project-root/
├── review-analyser-model.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/                 # Optional, if distributing saved model files
│   ├── sentiment_model.joblib
│   └── tfidf_vectorizer.joblib
├── data/                   # Optional; data/reviews.csv is displayed if present
│   └── reviews.csv
└── outputs/                # Optional saved evaluation artifacts
    ├── evaluation_report.txt
    └── confusion_matrix.png
```

## GitHub upload checklist

Upload the main Python application, this README, `requirements.txt`, `.gitignore`, and optionally the notebook if you want to showcase the ML workflow. Include model/data/output assets only when they are needed, safe to share, and appropriately licensed.

Do **not** upload `venv/`, `venv-1/`, `.venv/`, `__pycache__/`, or `.pyc` files.

## Attribution

If this project adapts code from another repository, retain appropriate attribution and follow the original repository's licence terms.
