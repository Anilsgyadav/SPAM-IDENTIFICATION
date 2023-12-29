import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pre-trained model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

st.title("Spam Identification")

input_sms = st.text_area("Type your message here")

if st.button('Predict'):
    try:
        # Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])

        # Use the pre-trained model to predict
        result = model.predict(vector_input)[0]

        # Display the prediction
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except Exception as e:
        st.error(f"Prediction error: {e}")


