import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load and combine dataset
true_file = "D://Job/My Projects/Fake-news-detection-using-NLP-main/data/fake.csv"
fake_file = "D://Job/My Projects/Fake-news-detection-using-NLP-main/data/true.csv"
if not (os.path.exists(true_file) and os.path.exists(fake_file)):
    print("Error: True.csv and/or Fake.csv not found in /content/.")
    raise FileNotFoundError("True.csv and Fake.csv must be in /content/.")

true_data = pd.read_csv(true_file)
fake_data = pd.read_csv(fake_file)
true_data['label'] = 0
fake_data['label'] = 1
data = pd.concat([true_data[['title', 'label']], fake_data[['title', 'label']]], ignore_index=True)

# Add synthetic fake titles
synthetic_fake_titles = pd.DataFrame({
    'title': [
        "Elon Musk Builds Moon Base by 2027",
        "SpaceX Discovers Alien Life on Mars",
        "Tesla CEO Announces Interstellar Travel by 2028",
        "NASA Hides Martian City Found by Musk",
        "Elon Musk’s Hyperloop Connects Earth to Mars"
    ],
    'label': [1] * 5
})
data = pd.concat([data, synthetic_fake_titles], ignore_index=True)

print(f"Dataset size: {len(data)} articles")
print("Class distribution:\n", data['label'].value_counts())

# Preprocess text
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['cleaned_text'] = data['title'].apply(clean_text)

# Prepare dataset for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['label'], test_size=0.2, random_state=42, stratify=data['label'])
train_dataset = NewsDataset(X_train.tolist(), y_train.tolist())
test_dataset = NewsDataset(X_test.tolist(), y_test.tolist())

# Load existing trained model
model_path = '/content/distilbert_fake_news_model_titles'
if os.path.exists(model_path):
    print(f"Loading trained model from {model_path}")
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
else:
    print("Model not found. Training from scratch...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Compute metrics
def compute_metrics(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions.argmax(axis=1)
    labels = eval_pred.label_ids
    f1 = f1_score(labels, predictions, average='weighted')
    return {'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_titles_finetune',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    logging_dir='./logs_titles_finetune',
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune
trainer.train()

# Evaluate
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Predict new headlines
def predict_news_distilbert(article_text):
    cleaned_article = clean_text(article_text)
    encoding = tokenizer(cleaned_article, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    prediction = outputs.logits.argmax().item()
    return 'Fake' if prediction == 1 else 'Real'

# Test headlines with Real/Fake labels
headlines = [
    "Rising Mortgage Rates and Recession Fears Stall a Fragile Housing Market",  # Real
    "Real Madrid Announce Trent Alexander-Arnold Signing from Liverpool",  # Real
    "Barcelona Sporting Director Deco Says Club Won’t Need to Sell Players This Summer",  # Real
    "How Real Is the India-Pakistan Nuclear War Threat?",  # Real
    "Coinbase Data Breach Tied to India-Based Outsourcing Partner TaskUs",  # Real
    "Mass Jailbreak in Karachi After Earthquake at Malir Jail",  # Real
    "Elon Musk Announces Plan to Colonize Mars by 2026",  # Fake
    "Secret Government Lab Discovers Time Travel Device",  # Fake
    "Scientists Discover New Species in Pacific Ocean",  # Real
    "Biden Announces New Climate Policy in 2025",  # Real
    "Aliens Invade New York with UFOs",  # Fake
    "Secret Vaccine Causes Superpowers in Children"  # Fake
]

print("\nTesting Headlines with DistilBERT:")
for headline in headlines:
    prediction = predict_news_distilbert(headline)
    print(f"Headline: {headline}\nPrediction: {prediction}\n")

# Save fine-tuned model
model.save_pretrained('/content/distilbert_fake_news_model_titles_finetuned')
tokenizer.save_pretrained('/content/distilbert_fake_news_tokenizer_titles_finetuned')
print("Model saved to /content/distilbert_fake_news_model_titles_finetuned")
