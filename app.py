import streamlit as st
import pickle
import gzip
import os

CLASSES = {0: "negative", 2: "neutral", 4: "positive"}

# Load trained model
@st.cache_resource
def load_model():
    model_path = "data/model.dat.gz"
    if not os.path.isfile(model_path):
        st.error("Model file not found at: data/model.dat.gz")
        return None
    with gzip.open(model_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

model = load_model()

# Streamlit UI
st.title("Sentiment Analysis")
st.markdown("Enter a sentence to predict whether the sentiment is *positive, **neutral, or **negative*.")

text_input = st.text_area("Input text:", height=150)

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    elif model is not None:
        try:
            x_vector = model.vectorizer.transform([text_input])
            prediction = model.predict(x_vector)
            sentiment = CLASSES.get(prediction[0], "unknown")

            if sentiment == "negative":
                st.markdown(f"""
                <div style="background-color:#ffcccc;padding:10px;border-radius:5px;color:#000;font-weight:bold;">
                Predicted Sentiment: {sentiment.capitalize()}
                </div>
                """, unsafe_allow_html=True)
            elif sentiment == "neutral":
                st.markdown(f"""
                <div style="background-color:#e6e6e6;padding:10px;border-radius:5px;color:#000;font-weight:bold;">
                Predicted Sentiment: {sentiment.capitalize()}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"Predicted Sentiment: *{sentiment.capitalize()}*")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
