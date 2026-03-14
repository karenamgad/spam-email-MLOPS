# src/predict.py
import joblib
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load saved model and vectorizer
clf = joblib.load('src/model.joblib')
vectorizer = joblib.load('src/vectorizer.joblib')

# Setup preprocessing
stemmer = PorterStemmer()
spam_important_words = {'you','now','free','win','winner','claim','prize','cash'}
stopwords_set = set(stopwords.words('english')) - spam_important_words

def classify_new_email(email_text):
    text = email_text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    X_email = vectorizer.transform([text])
    pred = clf.predict(X_email)[0]
    return "Spam" if pred==1 else "Ham"

# Example
if __name__ == "__main__":
    test_email = "Congratulations! You won a free ticket. Claim now."
    print("Prediction:", classify_new_email(test_email))