import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('wordnet') 
nltk.download('stopwords')

# Load your dataset
data = pd.read_csv("combined_data.csv")

# Inspect the dataset
print(data.head())
print(data.info())

# Define a function to preprocess the text data
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to form the cleaned text
    return " ".join(filtered_tokens)

# Apply preprocessing to the email texts
data['processed_text'] = data['text'].apply(preprocess_text)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data into numerical vectors
X = vectorizer.fit_transform(data['processed_text'])
y = data['label']  # Spam (1) or Not Spam (0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix to check True Positive, False Positive, True Negative, and False Negative
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

import joblib

# Save the model and vectorizer
joblib.dump(model, 'spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')


# Load the model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example: Classifying a new email
new_email = ["Congratulations! You've won a free ticket to Bahamas!"]
processed_email = preprocess_text(new_email[0])
email_vector = vectorizer.transform([processed_email])

# Make prediction
prediction = model.predict(email_vector)
if prediction == 1:
    print("Spam")
else:
    print("Not Spam")
