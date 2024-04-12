import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Load the model and vectorizer
model = pickle.load(open(r'C:\Users\debas\4th sem Python\SpamDetection\model2.pkl','rb'))
tfidf = pickle.load(open(r'C:\Users\debas\4th sem Python\SpamDetection\vectorizer.pkl','rb'))

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

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("ENTER SMS")

if st.button('PREDICT'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    # Convert the sparse matrix to a dense array
    vector_input_dense = vector_input.toarray()
    # Make prediction
    result = model.predict(vector_input_dense)[0]
    # Display the result
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
