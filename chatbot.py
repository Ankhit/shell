import json
import random

import joblib
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

# Prepare training data
X = []
y = []
for intent in intents_data:
    for pattern in intent['patterns']:
        X.append(preprocess_text(pattern))
        y.append(intent['tag'])

# Create TF-IDF vectorizer and transform data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Logistic Regression using GridSearchCV
param_grid_lr = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l2'],
}
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=3)  # Reduced cv to avoid warning
grid_search_lr.fit(X_train, y_train)
best_lr_model = grid_search_lr.best_estimator_

# Train Naive Bayes model with default parameters
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train Random Forest model with default parameters
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models on test data and collect metrics for visualization
models = [best_lr_model, nb_model, rf_model]
model_names = ['Logistic Regression', 'Naive Bayes', 'Random Forest']
accuracies = []
precisions = []
recalls = []

for model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

# Print evaluation metrics for each model
for name, acc, prec, rec in zip(model_names, accuracies, precisions, recalls):
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}\n")

# Save the best trained model and vectorizer
joblib.dump(best_lr_model, 'intents_model.joblib')
joblib.dump(vectorizer, 'intents_vectorizer.joblib')

print("Model and vectorizer saved successfully.")
