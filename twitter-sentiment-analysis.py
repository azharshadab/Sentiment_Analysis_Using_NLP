import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re

# Ensure NLTK resources are downloaded (do this once)
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing numbers
    text = re.sub(r'\d+', '', text)
    # Removing punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    words = word_tokenize(text)
    # Removing stopwords and applying stemming
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Read the data
tweeter_raw = pd.read_csv("tweeter_training.csv")

# Preprocessing the sentiment labels
tweeter_raw['Sentiment'] = tweeter_raw['Sentiment'].map({0: "Negative", 1: "Positive"}) 

# Initialize stemmer and stopwords
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

# Applying text preprocessing
tweeter_raw['SentimentText'] = tweeter_raw['SentimentText'].apply(preprocess_text)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(tweeter_raw['SentimentText'], tweeter_raw['Sentiment'], test_size=0.1, random_state=42)

# Vectorization
vectorizer = CountVectorizer(binary=True) 
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Training the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Predicting
y_pred = nb_classifier.predict(X_test_vectorized)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy}")

# This script is a direct implementation of the R code provided into Python, focusing on the key steps.

