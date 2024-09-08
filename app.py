import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import os

# Streamlit Interface configuration - must be at the very top
st.set_page_config(page_title="Mahabharata Q&A Chatbot", page_icon="🤖", layout="centered")

# Load Predefined Questions and Answers
@st.cache_data
def load_qa_pairs():
    with open('qa_pairs.json', 'r') as f:
        return json.load(f)

qa_pairs = load_qa_pairs()

# Load pre-trained model for sentence embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Get embeddings for all questions in the dataset
questions = [pair['question'] for pair in qa_pairs]
question_embeddings = model.encode(questions)

# Function to find the closest question
def get_best_answer(user_query):
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, question_embeddings)
    closest_index = np.argmax(similarities)
    return qa_pairs[closest_index]['answer']

# Function to translate text
def translate_text(text, target_language):
    return GoogleTranslator(source='auto', target=target_language).translate(text)

# Function to generate TTS audio and return base64 string
def generate_audio(text, language='en'):
    tts = gTTS(text, lang=language)
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
    os.remove("response.mp3")  # Clean up the file after use
    return base64.b64encode(audio_bytes).decode()

st.title("🤖 Mahabharata Q&A Chatbot")

languages = {"English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
language = st.selectbox("Select Language", list(languages.keys()), index=0)

# Input Box
user_input = st.text_input("Ask your question:")

if st.button("Submit"):
    if user_input:
        # Translate the input to English for model understanding
        user_input_translated = translate_text(user_input, 'en')

        # Get the answer in English
        response = get_best_answer(user_input_translated)

        # Translate the answer back to the selected language
        response_translated = translate_text(response, languages[language])
        
        # Display the answer with enhanced styling
        st.markdown(f"""
            <div style="background-color: #f9f9f9; padding: 20px; border-radius: 12px; box-shadow: 2px 2px 12px rgba(0,0,0,0.1);">
                <h3 style="color: #4c4c4c;">Answer:</h3>
                <p style="font-size: 18px; color: #333;">{response_translated}</p>
            </div>
        """, unsafe_allow_html=True)

        # Generate and display audio for the answer
        audio_data = generate_audio(response_translated, languages[language])
        audio_html = f'<audio controls><source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)

        # Option to download text and audio
        st.download_button('Download Answer Text', response_translated, file_name="answer.txt")
        st.download_button('Download Answer Audio', audio_data, file_name="answer.mp3")
        
        st.snow()
    else:
        st.warning("Please ask a question!")
