import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# Load dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "movie_genre_dataset.csv")

data = pd.read_csv(DATA_PATH)

X = data["description"]
y = data["genre"]
# Convert text to numeric features
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_vectorized, y)
# Streamlit UI
st.set_page_config(page_title="Movie Genre Classifier")
st.title("Movie Genre Classifier")
st.write("Enter a movie description to predict the genre")
user_input = st.text_area("Movie Description", height=150)
if st.button("Predict Genre"):
    if user_input.strip():
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)
        st.success(f"Predicted Genre: {prediction[0].capitalize()}")
    else:
        st.warning("Please enter a description.")
