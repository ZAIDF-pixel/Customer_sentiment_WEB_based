import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text=re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)