Fake News Detection with DistilBERT Model
This project uses a fine-tuned DistilBERT model to classify news headlines as "Real" or "Fake". It includes a Python script (evaluate_model.py) to evaluate model accuracy and a Streamlit app (app.py) for interactive predictions. The model was fine-tuned on a dataset of real and fake news headlines, achieving 99.47% accuracy locally (97% in Colab).
Project Overview

Objective: Detect fake news headlines using a fine-tuned DistilBERT model.

Label Mapping: 0 = Real, 1 = Fake.

Setup Instructions

Create Virtual Environment:
python -m venv venv
venv\Scripts\activate


Install Dependencies:
pip install transformers==4.52.2 torch==2.4.0 nltk==3.9.1 pandas==2.2.3 scikit-learn==1.5.2 safetensors==0.4.5 tqdm==4.66.5 streamlit==1.35.0


Verify Dependencies:
pip list | findstr "transformers torch nltk pandas scikit-learn safetensors tqdm streamlit"


Download NLTK Resources:
python -m nltk.downloader punkt punkt_tab stopwords wordnet


Configure VS Code (optional):

Install Python extension (Ctrl+Shift+X, search “Python”).
Select interpreter: D:\Projects\FakeNewsDetection\venv\Scripts\python.exe (Ctrl+Shift+P, “Python: Select Interpreter”).



Usage
1. Run model_training (Accuracy, Confusion Matrix, etc.) model_training.py

2. Evaluate Model Accuracy (evaluate_model.py)
Evaluates the model on a 20% test split (~4,496 samples) and tests a sample headline.

Run:cd D:\Projects\FakeNewsDetection
venv\Scripts\activate
python evaluate_model.py


Expected Runtime: ~8-8.5 minutes (CPU, 8-16 GB RAM), ~1-2 minutes (GPU).
Output Example:True.csv rows: 11240
Fake.csv rows: 11240
Predicting: 100%|██████████| 281/281 [08:14<00:00,  1.76s/it]
Accuracy: 0.9947
Precision: 0.9965
Recall: 0.9931
F1-Score: 0.9948

Predicting: 100%|██████████| 1/1 [00:01<00:00,  1.00it/s]
Headline: Scientists Discover New Species in Pacific Ocean
Cleaned Text: scientist discover new specie pacific ocean
Prediction: Real News



3. Run Streamlit App (app.py)
Interactive app to predict if headlines are “Real” or “Fake”.

Run:cd D:\Projects\FakeNewsDetection
venv\Scripts\activate
streamlit run app.py



Notes

Model Updates: evaluate_model.py and app.py perform inference only and do not update the model. To re-fine-tune in Colab:
Use a new save path:model.save_pretrained('/content/distilbert_fake_news_model_titles_finetuned_v2')


Save to Google Drive:from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/distilbert_fake_news_model_titles_finetuned /content/drive/MyDrive/



Dataset: Ensure True.csv and Fake.csv match Colab’s dataset for consistent accuracy.
GPU Acceleration: If available, CUDA reduces evaluate_model.py runtime to ~1-2 minutes:python -c "import torch; print(torch.cuda.is_available())"


Contributing

Report issues or suggest improvements by contacting the project maintainer.
Ensure dataset and model paths are updated for your environment.
