import joblib
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pypdf import PdfReader # Import PdfReader for PDF processing

# --- 1. Load the trained model, TF-IDF vectorizer, and Label Encoder ---
# Ensure these files exist from previous steps
try:
    loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    loaded_label_encoder = joblib.load('label_encoder.joblib')
    loaded_model = joblib.load('random_forest_model.joblib') # Load the Random Forest model
except FileNotFoundError as e:
    st.error(f"Error loading essential files: {e}. Please ensure 'tfidf_vectorizer.joblib', 'label_encoder.joblib', and 'random_forest_model.joblib' are available.")
    st.stop()

# --- 2. Preprocessing function (needs to be consistent with training) ---
# Ensure NLTK data is downloaded for the app environment
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_for_streamlit(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- 3. Streamlit Application UI and Logic ---
st.set_page_config(page_title="AI Resume Screener", layout="centered")

# Custom CSS for iOS-like styling
st.markdown(r'''
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600;700&display=swap');

    :root {
        --ios-blue: #007AFF;
        --ios-gray: #8E8E93;
        --ios-light-gray: #F2F2F7;
        --ios-dark-gray: #1C1C1E;
        --ios-separator: rgba(60, 60, 67, 0.36);
        --ios-background-dark: #000000;
        --ios-background-light: #FFFFFF;
    }

    body {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
        background-color: var(--ios-background-light);
        color: var(--ios-dark-gray);
    }

    .stApp {
        max-width: 700px;
        margin: auto;
        padding: 20px;
        background-color: var(--ios-background-light);
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }

    /* Header */
    .st-emotion-cache-18ni7ap.e1ewehp51 {
        background-color: transparent; /* No background for header */
    }
    h1 {
        color: var(--ios-dark-gray);
        font-weight: 700;
        font-size: 2.2em;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Subheader */
    h2 {
        color: var(--ios-dark-gray);
        font-weight: 600;
        font-size: 1.5em;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Text Area */
    .stTextArea > label {
        color: var(--ios-dark-gray);
        font-weight: 600;
        margin-bottom: 8px;
    }
    .stTextArea > div > div > textarea {
        border: 1px solid var(--ios-separator);
        border-radius: 8px;
        padding: 12px;
        font-size: 1em;
        line-height: 1.5;
        min-height: 200px; /* Make text area larger */
    }
    .stTextArea > div > div > textarea:focus {
        border-color: var(--ios-blue);
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.25);
    }

    /* Button */
    .stButton > button {
        background-color: var(--ios-blue);
        color: white;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 1.1em;
        font-weight: 600;
        border: none;
        transition: background-color 0.2s, transform 0.1s;
        width: 100%;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Metrics/Results */
    .stMetric > div > div:first-child {
        color: var(--ios-gray);
        font-weight: 600;
        font-size: 0.9em;
    }
    .stMetric > div > div:last-child > div:first-child {
        color: var(--ios-blue);
        font-weight: 700;
        font-size: 2em;
    }
    .stAlert {
        border-radius: 8px;
        background-color: var(--ios-light-gray);
        color: var(--ios-dark-gray);
        border: 1px solid var(--ios-separator);
    }
</style>
''', unsafe_allow_html=True)

st.title("AI Resume Screener")
st.write("Upload or paste a resume to predict its most suitable job category.")

uploaded_file = st.file_uploader("Upload a Resume (PDF only)", type=["pdf"])

# Text area for pasting resume text
resume_text_input = st.text_area("Or paste resume text here:", height=300)

processed_resume_text = ""
if uploaded_file is not None:
    # Process PDF
    try:
        processed_resume_text = extract_text_from_pdf(uploaded_file)
        st.info("Resume text extracted from PDF. You can now click 'Predict Category'.")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
elif resume_text_input:
    processed_resume_text = resume_text_input

# Prediction button
if st.button("Predict Category"):
    if processed_resume_text:
        # Preprocess the input text
        cleaned_input = preprocess_text_for_streamlit(processed_resume_text)

        # Transform using the loaded TF-IDF vectorizer
        input_vector = loaded_tfidf_vectorizer.transform([cleaned_input])

        # Make prediction using the loaded Random Forest model
        prediction_encoded = loaded_model.predict(input_vector)
        predicted_category = loaded_label_encoder.inverse_transform(prediction_encoded)

        st.success(f"Predicted Category: **{predicted_category[0]}**")

        st.subheader("How it works:")
        st.info("The AI model analyzes the content of the resume, identifying key skills, experiences, and keywords. It then matches these features against known job categories it was trained on to suggest the most fitting role. This helps in quickly filtering resumes for relevant positions.")

    else:
        st.warning("Please paste some resume text or upload a file to get a prediction.")