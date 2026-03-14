# src/api.py
from flask import Flask, request, jsonify
import joblib
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download stopwords (first run only)
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load saved model and vectorizer
clf = joblib.load("src/model.joblib")
vectorizer = joblib.load("src/vectorizer.joblib")

# Setup preprocessing
stemmer = PorterStemmer()
spam_important_words = {'you','now','free','win','winner','claim','prize','cash'}
stopwords_set = set(stopwords.words('english')) - spam_important_words

# Home route
@app.route("/")
def home():
    return "✅ Email Spam Classifier API is running! Use POST /predict to classify emails."

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email = data.get("email", "")
    
    # Preprocess
    text = email.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    
    # Transform & predict
    X_email = vectorizer.transform([text])
    pred = clf.predict(X_email)[0]
    label = "Spam" if pred == 1 else "Ham"
    
    return jsonify({"prediction": label})

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)