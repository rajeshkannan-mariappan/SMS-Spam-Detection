import streamlit as st
import joblib
import re
import string

# Load trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean text messages
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to predict SMS spam or ham
def predict_sms(message):
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(vectorized_message)[0]
    return "Spam" if prediction == 1 else "Ham"

# Streamlit UI
st.title("📩 SMS Spam Classifier")
st.write("Enter a text message to check if it's **Spam or Ham**.")

user_input = st.text_area("Enter SMS Message Here:", "")

if st.button("Check"):
    if user_input.strip():
        result = predict_sms(user_input)
        if result == "Spam":
            st.error("🚨 **Spam Message Detected!**")
        else:
            st.success("✅ **Safe Message (Ham)**")
    else:
        st.warning("⚠ Please enter a message to classify.")
