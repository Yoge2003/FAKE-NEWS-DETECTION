import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    st.warning("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess text (match Colab)
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    cleaned = ' '.join(tokens)
    return cleaned

# Load fine-tuned model and tokenizer
model_path = "D:/Projects/FakeNewsDetection/models/distilbert_fake_news_model_titles_finetuned"
tokenizer_path = "D:/Projects/FakeNewsDetection/models/distilbert_fake_news_tokenizer_titles_finetuned"

# Verify model files
if not os.path.exists(model_path):
    st.error(f"Model directory not found at {model_path}. Please check the path or download the model.")
    st.stop()
if not os.path.exists(os.path.join(model_path, 'config.json')):
    st.error(f"config.json missing at {model_path}. Re-download the model from Colab.")
    st.stop()
if not (os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or os.path.exists(os.path.join(model_path, 'model.safetensors'))):
    st.error(f"Neither pytorch_model.bin nor model.safetensors found at {model_path}. Re-download the model from Colab.")
    st.stop()

# Verify tokenizer files
if not os.path.exists(tokenizer_path):
    st.error(f"Tokenizer directory not found at {tokenizer_path}. Please check the path or download the tokenizer.")
    st.stop()
if not os.path.exists(os.path.join(tokenizer_path, 'vocab.txt')):
    st.error(f"vocab.txt missing at {tokenizer_path}. Re-download the tokenizer from Colab.")
    st.stop()

try:
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.eval()
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}. Ensure all files are present and uncorrupted.")
    st.stop()

# Prediction function
def predict_news_distilbert(headline):
    cleaned_text = clean_text(headline)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Fixed dictionary comprehension
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    # Debugging logs
    st.write(f"**Debug**: Cleaned Text: {cleaned_text}")
    st.write(f"**Debug**: Logits: {logits.tolist()[0]}")
    return "Fake News" if prediction == 1 else "Real News"

# Streamlit UI
st.title("📰 Fake News Detection")
st.write("Paste a news headline below to check if it's Real or Fake News.")

# Input headline
input_text = st.text_area("Enter Headline", value="", height=100)

# Predict button
if st.button("Predict"):
    if input_text.strip():
        result = predict_news_distilbert(input_text)
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("⚠️ Please enter a headline.")