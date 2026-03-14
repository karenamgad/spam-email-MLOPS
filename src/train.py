# src/train.py
from preprocess import preprocess_dataset, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load and preprocess dataset
df = preprocess_dataset('data/spam_ham_dataset.csv')

# Features and labels
corpus = df['processed_text'].tolist()
y = df['label_num']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
X = vectorizer.fit_transform(corpus)

# Split train-test
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = MultinomialNB()
clf.fit(x_train, y_train)

# Evaluate
y_pred = clf.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer as .joblib
joblib.dump(clf, 'src/model.joblib')
joblib.dump(vectorizer, 'src/vectorizer.joblib')
print("Model and vectorizer saved in src/")