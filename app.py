import json
import random

import joblib
import nltk
import streamlit as st

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

try:
    model = joblib.load('intents_model.joblib')
    vectorizer = joblib.load('intents_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please run chatbot.py first to train the model.")
    st.stop()

try:
    with open('intents.json', 'r') as file:
        intents_data = json.load(file)
except FileNotFoundError:
    st.error("intents.json file not found.")
    st.stop()

def get_response(tag):
    for intent in intents_data:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "I'm not sure how to respond to that."

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

st.title("Intent-Based Chatbot")

for message in st.session_state.conversation:
    st.text(message)

user_input = st.text_input("You: ", key="user_input")

if st.button("Send"):
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        user_input_vectorized = vectorizer.transform([preprocessed_input])
        
        predicted_intent = model.predict(user_input_vectorized)[0]
        
        response = get_response(predicted_intent)
        
        st.session_state.conversation.append(f"You: {user_input}")
        st.session_state.conversation.append(f"Chatbot: {response}")
        
        st.rerun()

if st.button("Clear Conversation"):
    st.session_state.conversation.clear()
    st.rerun()
