import os
import io
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="wide",
)

STOPWORDS = {
    "a", "an", "the", "is", "it", "this", "that", "i", "you", "he", "she",
    "we", "they", "am", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "for", "with", "and", "or", "so", "as",
    "my", "your", "his", "her", "its", "our", "their", "me", "him", "them",
    "us", "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "just",
}

def clean_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text: str) -> str:
    return " ".join(w for w in clean_text(text).split() if w not in STOPWORDS)

def generate_default_data():
    subjects = [
        "the product", "this movie", "the restaurant", "the phone", "the laptop",
        "this book", "the service", "the app", "the hotel room", "the course",
        "this game", "the headphones", "the software", "the delivery", "the show",
        "this camera", "the customer support", "the coffee", "this album", "the shoes"
    ]
    pos = [
        "I absolutely love {s}, it exceeded all my expectations.",
        "{s} works perfectly and the quality is outstanding.",
        "Really impressed with {s}, would highly recommend it to anyone.",
        "{s} is fantastic, worth every penny.",
        "What a great experience with {s}, everything was smooth and enjoyable.",
        "{s} made my day, I'm extremely satisfied.",
        "Excellent! {s} is exactly what I was hoping for.",
        "I'm so happy with {s}, it's the best I've tried so far.",
        "{s} is amazing and I will definitely buy/use it again.",
        "Superb quality, {s} really delivers on its promises.",
        "{s} is wonderful, brilliant, and simply a joy to use.",
        "Loved every bit of {s}, five stars without a doubt.",
    ]
    neg = [
        "I hate {s}, it was a complete waste of money.",
        "{s} broke down within days, extremely disappointing.",
        "Terrible experience with {s}, I would not recommend it.",
        "{s} is awful and nothing like what was advertised.",
        "Very frustrated with {s}, it simply doesn't work as expected.",
        "{s} was a huge letdown, poor quality all around.",
        "I regret buying/using {s}, total disaster.",
        "{s} is the worst I've ever encountered, avoid at all costs.",
        "Horrible! {s} left me extremely unsatisfied.",
        "{s} kept failing and customer support was useless.",
        "Such a bad experience, {s} does not live up to the hype.",
        "Disgusted with {s}, it was slow, buggy, and unreliable.",
    ]
    neu = [
        "{s} was okay, nothing special but not bad either.",
        "{s} is average, does the job but could be improved.",
        "I have mixed feelings about {s}, some parts were good, others not.",
        "{s} is fine for the price, though there's room for improvement.",
        "Not sure how I feel about {s}, it was just a typical experience.",
        "{s} met basic expectations, nothing more, nothing less.",
        "{s} was decent overall, a fairly standard experience.",
        "It's an ordinary experience with {s}, neither great nor terrible.",
        "{s} works as described, no complaints but no praise either.",
        "{s} is acceptable, though I probably won't go out of my way for it again.",
        "A middle-of-the-road experience with {s}.",
        "{s} was neither impressive nor disappointing, just average.",
    ]
    rows = []
    for s in subjects:
        for t in pos: rows.append((t.format(s=s), "positive"))
        for t in neg: rows.append((t.format(s=s), "negative"))
        for t in neu: rows.append((t.format(s=s), "neutral"))
    import random
    random.seed(42)
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "sentiment"])

# -----------------------------------------------------------------------------
# Model & Asset Loading (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading...")
def load_assets():
    model_path = os.path.join(MODELS_DIR, "sentiment_model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer

    # Auto-train fallback in ~0.5s if model folder was not uploaded
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    df = generate_default_data()
    clean_series = df["text"].apply(preprocess)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_vec = vectorizer.fit_transform(clean_series)
    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_vec, df["sentiment"])
    return model, vectorizer


def predict_sentiment(text: str):
    """Clean text, vectorize, and predict sentiment with probabilities."""
    clean = preprocess(text)
    if not clean.strip():
        clean = text.lower().strip()

    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    prob_dict = {
        cls_name: float(prob)
        for cls_name, prob in zip(model.classes_, proba)
    }
    return pred, prob_dict, clean



model, vectorizer = load_assets()

st.title("Customer Review Sentiment Analyzer")
st.write("Classify a review as positive, neutral, or negative.")

tab_predict, tab_batch, tab_info = st.tabs(
    ["Single Review", "CSV Batch Analysis", "Model & Evaluation"]
)

with tab_predict:
    review = st.text_area(
        "Enter a customer review",
        placeholder="Example: The product works well and delivery was quick.",
        height=120,
        key="single_review",
    )
    if st.button("Analyze sentiment"):
        if not review.strip():
            st.warning("Enter a review before analyzing.")
        else:
            sentiment, probabilities, cleaned_text = predict_sentiment(review)
            st.subheader(f"Predicted sentiment: {sentiment.title()}")
            st.write("Estimated class probabilities")
            for label in ("positive", "neutral", "negative"):
                st.write(f"{label.title()}: {probabilities.get(label, 0.0) * 100:.1f}%")
            with st.expander("Show processed text"):
                st.write(cleaned_text)

with tab_batch:
    st.write("Upload a CSV file containing a column of review text.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if df.empty:
                st.warning("The CSV file has no rows.")
            else:
                text_column = st.selectbox("Select the review-text column", options=list(df.columns))
                if st.button("Analyze CSV"):
                    result = df.copy()
                    predictions = []
                    probability_rows = []
                    for value in result[text_column].fillna("").astype(str):
                        if value.strip():
                            label, probs, _ = predict_sentiment(value)
                        else:
                            label, probs = "empty", {}
                        predictions.append(label)
                        probability_rows.append(probs)
                    result["predicted_sentiment"] = predictions
                    for label in ("positive", "neutral", "negative"):
                        result[f"probability_{label}"] = [
                            row.get(label, None) for row in probability_rows
                        ]
                    st.subheader("Prediction summary")
                    st.dataframe(result.head(100), use_container_width=True)
                    counts = pd.Series(predictions).value_counts()
                    st.bar_chart(counts)
                    st.download_button(
                        "Download results as CSV",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

with tab_info:
    st.subheader("Model overview")
    st.write("Feature extraction: TF-IDF ")
    st.write("Classifier: Logistic Regression")
    st.write("Classes: positive, neutral, negative.")

    st.subheader("Demonstration evaluation on synthetic data")

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    eval_df = generate_default_data()
    train_df, test_df = train_test_split(
        eval_df, test_size=0.25, random_state=42, stratify=eval_df["sentiment"]
    )
    eval_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = eval_vectorizer.fit_transform(train_df["text"].apply(preprocess))
    X_test = eval_vectorizer.transform(test_df["text"].apply(preprocess))
    eval_model = LogisticRegression(max_iter=1000, C=5.0)
    eval_model.fit(X_train, train_df["sentiment"])
    y_pred = eval_model.predict(X_test)
    y_true = test_df["sentiment"]

    st.metric("Synthetic hold-out accuracy", f"{accuracy_score(y_true, y_pred) * 100:.1f}%")
    st.text("Classification report (synthetic hold-out split):")
    st.text(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["negative", "neutral", "positive"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Synthetic Hold-out Split")
    st.pyplot(fig)
    plt.close(fig)

    report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")
    matrix_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    if os.path.exists(report_path):
        st.subheader("Additional saved evaluation report")
        with open(report_path, "r", encoding="utf-8") as f:
            st.text(f.read())
    if os.path.exists(matrix_path):
        st.subheader("Additional saved confusion matrix")
        st.image(matrix_path, use_container_width=True)

    data_path = os.path.join(DATA_DIR, "reviews.csv")
    if os.path.exists(data_path):
        st.subheader("Available review data")
        try:
            sample_df = pd.read_csv(data_path)
            st.dataframe(sample_df.head(20), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not load data/reviews.csv: {exc}")