# -*- coding: utf-8 -*-
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('intern chatbot - myquiz.csv')  # Adjust path for Colab

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# List of models to test
models = [
    "facebook/bart-large-mnli", #45%
    "joeddav/xlm-roberta-large-xnli", # 50%
    "typeform/distilbert-base-uncased-mnli", #fast, no good
    "valhalla/distilbart-mnli-12-3", # always guesses 1
    "microsoft/deberta-xlarge-mnli" #always guesses 1
]

# Shortened skill names for plotting
skill_names = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

# Detailed skill descriptions for classification
skills = [
    "Remember: Identify or recall facts and basic concepts. For example, 'What is the capital of Thailand?'",
    "Understand: Explain ideas or concepts in your own words. This should not just be repeating a concept. For example, 'Summarize the main idea of a given text.'",
    "Apply: Use information in new situations or solve problems using knowledge and skills. For example, 'ในกรณีที่ผู้ใช้ถอนความยินยอมในการใช้ข้อมูลส่วนบุคคล ผู้พัฒนา AI ควรทำอย่างไร? ก. เก็บข้อมูลต่อไปแต่ไม่ใช้งาน ข. ขายข้อมูลให้กับบริษัทอื่น ค. หยุดการใช้ข้อมูลและลบข้อมูลทันที ง. ใช้ข้อมูลต่อไปแม้ผู้ใช้จะถอนความยินยอม'",
    "Analyze: Break down information into parts to explore relationships and patterns. For example, 'Compare and contrast the causes of World War I and World War II.'",
    "Evaluate: Make judgments based on criteria and standards. For example, 'Assess the effectiveness of a new teaching method compared to traditional methods.'",
    "Create: Produce new or original work by combining elements in a novel way. For example, 'Design an experiment to test the effects of sunlight on plant growth.'"
]

# Function to classify a question
def classify_question(classifier, question):
    result = classifier(question, candidate_labels=skills, multi_label=True)
    # Convert the result to a dictionary with skill names and scores
    classification = {label.split(":")[0]: score for label, score in zip(result["labels"], result["scores"])}
    return classification

# Function to evaluate a model
def evaluate_model(model_name):
    print(f"Evaluating model: {model_name}")
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    total_loss = 0
    all_true_labels = []
    all_predicted_labels = []
    
    for i, question in enumerate(data['question']):
        classification = classify_question(classifier, question)
        pred = [classification.get(skill.split(":")[0], 0) for skill in skills]
        
        # Convert to tensor and move to GPU
        true_labels_tensor = torch.tensor([true_labels[i]], dtype=torch.float32).to(device)
        predictions_tensor = torch.tensor([pred], dtype=torch.float32).to(device)
        
        # Define the loss function (Binary Cross-Entropy for multi-label classification)
        loss_function = torch.nn.BCELoss()
        
        # Calculate the loss
        loss = loss_function(predictions_tensor, true_labels_tensor)
        total_loss += loss.item()
        
        # Store true labels and predicted labels for later evaluation
        all_true_labels.append(true_labels_tensor.cpu().numpy())
        all_predicted_labels.append((predictions_tensor.cpu().numpy() > 0.5).astype(float))
        
        # Print results for the current question
        print(f"Question {i+1}/{len(data)}")
        print(f"Loss: {loss.item()}")
        print(f"True Labels: {true_labels_tensor.cpu().numpy()}")
        print(f"Predictions: {predictions_tensor.cpu().numpy()}")
    
    # Flatten lists for metric calculation
    all_true_labels = np.vstack(all_true_labels)
    all_predicted_labels = np.vstack(all_predicted_labels)
    
    # Calculate accuracy
    total_true = all_true_labels.sum()
    total_pred = all_predicted_labels.sum()
    correct_preds = (all_true_labels == all_predicted_labels).sum()
    accuracy = correct_preds / (all_true_labels.size)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predicted_labels, average='micro', zero_division=0)
    
    print(f"Average Loss: {total_loss / len(data)}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("-" * 50)
    
    # Plot combined confusion matrix for all skills
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, skill in enumerate(skill_names):
        cm = confusion_matrix(all_true_labels[:, i], all_predicted_labels[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not '+skill, skill], yticklabels=['Not '+skill, skill], ax=axes[i])
        axes[i].set_title(f'Confusion Matrix for {skill}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace("/", "_")}.png')
    plt.show()

# Load ground truth labels
true_labels = data[['ind-remember', 'ind-understand', 'ind-apply', 'ind-analyze', 'ind-evaluate', 'ind-create']].values

# Evaluate all models
for model in models:
    evaluate_model(model)
