# -*- coding: utf-8 -*-
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure the correct version of setuptools

# Load the dataset
data = pd.read_csv('intern chatbot - myquiz.csv')  # Adjust path for Colab

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define Bloom's Taxonomy skills
skills = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

# Add labels column to datasets
train_data['labels'] = train_data[['ind-remember', 'ind-understand', 'ind-apply', 'ind-analyze', 'ind-evaluate', 'ind-create']].values.tolist()
test_data['labels'] = test_data[['ind-remember', 'ind-understand', 'ind-apply', 'ind-analyze', 'ind-evaluate', 'ind-create']].values.tolist()

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = self.data.iloc[index]['question']
        labels = self.data.iloc[index]['labels']
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Create dataset and dataloaders
train_dataset = CustomDataset(train_data, tokenizer, max_len=256)
test_dataset = CustomDataset(test_data, tokenizer, max_len=256)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(skills)).to(device)

# Define training arguments without tensorboard logging
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"  # Disable reporting to avoid tensorboard issue
)

# Define compute_metrics function
def compute_metrics(p):
    logits, labels = p
    predictions = (logits > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Validation Loss: {eval_results['eval_loss']}")
print(f"Validation Accuracy: {eval_results['eval_accuracy']}")
print(f"Validation Precision: {eval_results['eval_precision']}")
print(f"Validation Recall: {eval_results['eval_recall']}")
print(f"Validation F1 Score: {eval_results['eval_f1']}")

# Save the model
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
