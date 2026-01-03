# evaluation model
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
from tqdm import tqdm

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
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
    return ' '.join(tokens)

# Load model and tokenizer
model_path = "D:/Job/My Projects/Fake-news-detection-using-NLP-main/models/distilbert_fake_news_model_titles_finetuned/model.safetensors"
tokenizer_path = "D:/Job/My Projects/Fake-news-detection-using-NLP-main/models/distilbert_fake_news_tokenizer_titles_finetuned"

# Verify files
if not os.path.exists(model_path):
    print(f"Error: Model directory not found at {model_path}")
    exit()
if not os.path.exists(os.path.join(model_path, 'config.json')):
    print(f"Error: config.json missing at {model_path}")
    exit()
if not (os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or os.path.exists(os.path.join(model_path, 'model.safetensors'))):
    print(f"Error: Neither pytorch_model.bin nor model.safetensors found at {model_path}")
    exit()
if not os.path.exists(os.path.join(tokenizer_path, 'vocab.txt')):
    print(f"Error: vocab.txt missing at {tokenizer_path}")
    exit()

try:
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load dataset
try:
    true_data = pd.read_csv('D:/Projects/FakeNewsDetection/data/True.csv')
    fake_data = pd.read_csv('D:/Projects/FakeNewsDetection/data/Fake.csv')
except FileNotFoundError as e:
    print(f"Error: Dataset not found: {e}")
    exit()

true_data['label'] = 0  # Real
fake_data['label'] = 1  # Fake
data = pd.concat([true_data[['title', 'label']], fake_data[['title', 'label']]], ignore_index=True)

# Preprocess titles
data['cleaned_text'] = data['title'].apply(clean_text)

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prediction function with batching
def predict_batch(texts, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(batch_preds)
        # Clear memory
        del inputs, outputs, logits, batch_preds
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return predictions

# Evaluate on test set
test_texts = test_data['cleaned_text'].tolist()
test_labels = test_data['label'].tolist()
predictions = predict_batch(test_texts, batch_size=16)

# Compute metrics
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='binary')
recall = recall_score(test_labels, predictions, average='binary')
f1 = f1_score(test_labels, predictions, average='binary')

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
