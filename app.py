import base64
import io
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, render_template, request
from groq import Groq

from nltk_processing import preprocess_text


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "sentiment.db"

model = joblib.load(BASE_DIR / "sentiment_model.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = Flask(__name__)


def get_reviews():
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        return pd.read_sql_query("SELECT * FROM reviews ORDER BY id DESC", connection)
    finally:
        connection.close()


def initialize_database():
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review TEXT,
                sentiment TEXT,
                date_time TEXT
            )
        """)
        connection.commit()
    finally:
        connection.close()


initialize_database()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/dashboard")
def dashboard():
    reviews = get_reviews()
    counts = reviews["sentiment"].value_counts() if not reviews.empty else {}
    recent_reviews = reviews.head(6).fillna("").to_dict(orient="records")
    return jsonify({
        "total": len(reviews),
        "counts": {
            "positive": int(counts.get("positive", 0)),
            "negative": int(counts.get("negative", 0)),
            "neutral": int(counts.get("neutral", 0)),
        },
        "recent": recent_reviews,
    })


@app.post("/api/analyze")
def analyze_review():
    payload = request.get_json(silent=True) or {}
    review = str(payload.get("review", "")).strip()
    if not review:
        return jsonify({"error": "Enter a customer review first."}), 400

    cleaned_review = preprocess_text(review)
    review_tfidf = tfidf.transform([cleaned_review])
    prediction = str(model.predict(review_tfidf)[0])
    probabilities = model.predict_proba(review_tfidf)[0]
    confidence = float(max(probabilities) * 100)

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute(
            "INSERT INTO reviews (review, sentiment, date_time) VALUES (?, ?, ?)",
            (review, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        connection.commit()
    finally:
        connection.close()

    return jsonify({"sentiment": prediction, "confidence": confidence})


@app.get("/api/visualizations")
def visualizations():
    reviews = get_reviews()
    if reviews.empty:
        return jsonify({"error": "Analyze some reviews before viewing trends."}), 400

    sentiments = ["positive", "negative", "neutral"]
    counts = reviews["sentiment"].value_counts()
    values = [int(counts.get(sentiment, 0)) for sentiment in sentiments]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].bar(sentiments, values, color=["#16846b", "#d6534d", "#d7a63e"])
    axes[0, 0].set_title("Sentiment Distribution")
    axes[0, 0].set_xlabel("Sentiment")
    axes[0, 0].set_ylabel("Number of Reviews")

    axes[0, 1].pie(values, labels=sentiments, autopct="%1.1f%%")
    axes[0, 1].set_title("Sentiment Percentage")

    cumulative = [sum(values[:index + 1]) for index in range(len(values))]
    axes[1, 0].plot(sentiments, cumulative, marker="o", color="#167f8b")
    axes[1, 0].set_title("Cumulative Sentiment Count")
    axes[1, 0].set_xlabel("Sentiment")
    axes[1, 0].set_ylabel("Count")

    review_lengths = reviews["review"].fillna("").str.len()
    sentiment_numbers = reviews["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
    axes[1, 1].scatter(review_lengths, sentiment_numbers, color="#d6534d", alpha=0.7)
    axes[1, 1].set_title("Review Length vs Sentiment")
    axes[1, 1].set_xlabel("Review Length")
    axes[1, 1].set_ylabel("Sentiment")
    axes[1, 1].set_yticks([0, 1, 2], ["Negative", "Neutral", "Positive"])

    figure.tight_layout()
    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=140)
    plt.close(figure)
    encoded_image = base64.b64encode(image.getvalue()).decode("ascii")
    return jsonify({"image": f"data:image/png;base64,{encoded_image}"})


@app.post("/api/summary")
def generate_summary():
    if client is None:
        return jsonify({"error": "Set the GROQ_API_KEY environment variable to enable summaries."}), 503

    reviews = get_reviews()
    if reviews.empty:
        return jsonify({"error": "Analyze some reviews before generating a summary."}), 400

    positive = int((reviews["sentiment"] == "positive").sum())
    negative = int((reviews["sentiment"] == "negative").sum())
    neutral = int((reviews["sentiment"] == "neutral").sum())
    review_text = "\n".join(reviews["review"].fillna("").tolist()[:50])
    prompt = f"""
You are a customer feedback analyst.

Analyze the following customer feedback.

Positive reviews: {positive}
Negative reviews: {negative}
Neutral reviews: {neutral}

Customer reviews:
{review_text}

Provide a simple professional summary containing:

1. Overall customer sentiment
2. Positive feedback
3. Negative feedback
4. Common customer problems
5. Suggestions for improvement

Keep the answer concise and easy to understand.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return jsonify({"summary": response.choices[0].message.content})
    except Exception:
        app.logger.exception("Groq summary request failed")
        return jsonify({"error": "The summary service could not complete the request."}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
