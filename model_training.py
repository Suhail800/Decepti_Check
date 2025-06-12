# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import load_data, clean_text # Import from utils.py

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased"
SAVE_MODEL_PATH = "models/distilbert_deception_model"
BATCH_SIZE = 16
NUM_EPOCHS = 3 # You can increase this for better performance, but more computation
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Ensure model save path exists ---
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

def train_model():
    # 1. Load Data
    df = load_data()
    
    # 2. Preprocess Data
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Map labels to integers: 0 for truthful, 1 for deceptive
    df['label_id'] = df['label'].apply(lambda x: 0 if x == 'truthful' else 1)
    
    # 3. Split Data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['cleaned_text'].tolist(),
        df['label_id'].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label_id'] # Ensure balanced split
    )
    
    # 4. Load Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # 5. Tokenize Data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    
    # Create Dataset objects for Hugging Face Trainer
    class DeceptionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = DeceptionDataset(train_encodings, train_labels)
    val_dataset = DeceptionDataset(val_encodings, val_labels)

    # 6. Load Model
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 7. Define Metrics for Evaluation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # 8. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=NUM_EPOCHS,     # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,   # batch size per device during evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        eval_strategy="epoch",     # Evaluate at the end of each epoch
        save_strategy="epoch",           # Save model at the end of each epoch
        load_best_model_at_end=True,     # Load the best model at the end of training
        metric_for_best_model="f1",      # Metric to use to compare models
        report_to="none"                 # Disable integrations like W&B for simplicity
    )

    # 9. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 10. Train Model
    print("Starting model training...")
    trainer.train()
    print("Training finished!")

    # 11. Save Model and Tokenizer
    model.save_pretrained(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_MODEL_PATH)
    print(f"Model and tokenizer saved to {SAVE_MODEL_PATH}")

if __name__ == '__main__':
    train_model()