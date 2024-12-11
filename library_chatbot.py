import nltk
import json
import random
import ssl
import streamlit as st
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import numpy as np

# Disable SSL verification (for environments with SSL issues)
try:
    _create_default_https_context = ssl._create_default_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = ssl._create_unverified_context

# Load intent data from intent.json
with open("intent.json") as file:
    data = json.load(file)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess data
def preprocess_data(data):
    patterns = []
    responses = []
    intents = []

    for intent in data['intents']:
        # Access the 'tag' key instead of 'intent'
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            lemmatized_pattern = ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_list])  # Join words into a string
            patterns.append(lemmatized_pattern)  # Append the string, not the list
            intents.append(intent['tag'])  # Use 'tag' here instead of 'intent'
        responses.append(intent['responses'])
    
    return patterns, intents, responses


# Process the patterns and intents
patterns, intents, responses = preprocess_data(data)
print(patterns)

# Create a TF-IDF Vectorizer and transform the patterns
vectorizer = CountVectorizer(tokenizer=lambda sentence: sentence.split(), stop_words=None)
X = vectorizer.fit_transform(patterns)

# Train the model using Support Vector Classifier (SVC)
model = SVC(kernel='linear')
model.fit(X, intents)


# Function to predict intent
def predict_intent(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    lemmatized_sentence = [lemmatizer.lemmatize(w) for w in sentence_words]
    vector = vectorizer.transform([' '.join(lemmatized_sentence)])  # Join the lemmatized words
    return model.predict(vector)[0]

# Function to get a response
def get_response(intent):
    for i in range(len(data['intents'])):
        if data['intents'][i]['tag'] == intent:  # Use 'tag' here instead of 'intent'
            return random.choice(data['intents'][i]['responses'])
    return "Sorry, I don't understand."

# Streamlit Web Interface
def run_chatbot():
    st.title("Library Chatbot")
    
    user_input = st.text_input("You:", "Hi")
    
    if user_input:
        intent = predict_intent(user_input)
        bot_response = get_response(intent)
        st.write(f"Bot: {bot_response}")

# Run the Streamlit app
if __name__ == "__main__":
    run_chatbot()
