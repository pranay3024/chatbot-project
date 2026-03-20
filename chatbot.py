import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ data
faq = [
    ("What is your refund policy?", "We offer 7-day refund."),
    ("How long is delivery?", "Delivery takes 3-5 days."),
    ("Do you offer COD?", "Yes, Cash on Delivery is available.")
]

questions = [q[0] for q in faq]
answers = [q[1] for q in faq]

# Vectorize
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, question_vectors)
    index = similarity.argmax()
    return answers[index]

# UI
st.title("🛒 Customer Support Chatbot")

user_input = st.text_input("Ask your question")

if user_input:
    response = get_answer(user_input)
    st.write("🤖:", response)