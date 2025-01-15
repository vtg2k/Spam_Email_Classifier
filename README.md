 Spam Email Classifier

## Overview
This project aims to build a machine learning model to classify emails as "spam" or "not spam" using Natural Language Processing (NLP) techniques. The model is trained using a labeled dataset of emails and can predict whether an email is spam based on its content.

## Technologies Used
- **Python**: Programming language used for building the model.
- **Scikit-learn**: Machine learning library for building and training the classification model.
- **Naive Bayes**: The classification algorithm used for the email spam classification.
- **TF-IDF**: A technique to convert text into numerical vectors.
- **NLTK**: A Natural Language Processing toolkit used for text preprocessing (like tokenizing, removing stopwords).

## How It Works
1. **Preprocessing**: The text data from emails is cleaned, tokenized, and stopwords are removed.
2. **Feature Extraction**: The emails are transformed into numerical representations using TF-IDF.
3. **Training**: A Naive Bayes classifier is trained on the processed data.
4. **Prediction**: The trained model can predict whether a new email is spam or not.

## Dataset
- The dataset used for this project contains labeled emails, with each email marked as "spam" or "ham" (non-spam). The dataset can be found at: [Spam Email Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset).
