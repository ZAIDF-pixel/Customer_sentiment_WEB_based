# Review Signal

A local web dashboard for customer sentiment analysis, review history, visualizations, and AI-generated feedback summaries.

## Run locally on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python app.py
```

Open http://127.0.0.1:5000 in a browser. The included model and vectorizer are used for predictions. Set `GROQ_API_KEY` in the environment to enable AI summaries; do not put the key in source files.

## Training data

`model.py` can train new model artifacts when `Customer_Sentiment.csv` is present in the project root with `review_text` and `sentiment` columns. The dataset is intentionally excluded from Git because it contains customer review text. The app creates `sentiment.db` locally to store analyzed reviews; that database is also excluded.