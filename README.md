import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('movies.csv')

# Preprocess the text (cleaning and tokenization)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['plot'] = data['plot'].apply(preprocess_text)

# Splitting the data
X = data['plot']
y = data['genre'].apply(lambda x: x.split('|'))  # Assuming genres are separated by '|'

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

# Creating pipelines for different classifiers
nb_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', MultinomialNB())
])

lr_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', LogisticRegression(max_iter=1000))
])

svm_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('clf', SVC(kernel='linear'))
])

# Training the models
nb_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Evaluating the models
nb_pred = nb_pipeline.predict(X_test)
lr_pred = lr_pipeline.predict(X_test)
svm_pred = svm_pipeline.predict(X_test)

# Classification reports
print("Naive Bayes Classifier:\n", classification_report(y_test, nb_pred, target_names=mlb.classes_))
print("Logistic Regression Classifier:\n", classification_report(y_test, lr_pred, target_names=mlb.classes_))
print("SVM Classifier:\n", classification_report(y_test, svm_pred, target_names=mlb.classes_))

# Accuracy scores
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
