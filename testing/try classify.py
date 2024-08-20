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

# Select the first 3 rows
# data = data.iloc[:3, :]

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# List of models to test
models = [
    "facebook/bart-large-mnli"
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


#evaluate for everything

# evaluation shoud only be applied for choosing between options explicitly, when comparing whether something is between two options. If it needs to be infered on what is an outcome, it would be moreso analysing rather than evaluating

#creating, unless the question explicitly states to think of an idea, this should be false

#analysing, can just be digesting an issue or considering a specific outcome of a concept. Specifically if the question contains 'how' or 'why' this is true

#applying, this should only be true if a concept is used in a unique context, separate from the original definition. Unless a specific scenario is stated this is likely false.

#remerming, this shoud only be true if a techinical concept or terminoloy is mentioned, anything general knowlege doens't count.

# Detailed skill descriptions for classification
skills = [
    "Remember: Recall basic facts and concepts.",
    "Understand: Explain ideas or concepts in your own words without just repeating information. For example, summarize or paraphrase a given text.",
    "Apply: Use information in new situations or solve problems in unique contexts.",
    "Analyze: Break down information into parts to explore relationships, e.g., answer 'how' or 'why' questions.",
    "Evaluate: Make judgments based on criteria and standards through checking and critiquing. This does require some amount of thinking to be considered true",
    "Create: Produce new or original work by combining elements in a novel way, e.g., design an experiment."
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
    predictions = []

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
        all_predicted_labels.append((predictions_tensor.cpu().numpy() > 0.4).astype(float))
        predictions.append((predictions_tensor.cpu().numpy() > 0.5).astype(int))

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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not ' + skill, skill],
                    yticklabels=['Not ' + skill, skill], ax=axes[i])
        axes[i].set_title(f'Confusion Matrix for {skill}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    # plt.show()

    return predictions

# Load ground truth labels
true_labels = data[['ind-remember', 'ind-understand', 'ind-apply', 'ind-analyze', 'ind-evaluate', 'ind-create']].values

# Evaluate all models
for model in models:
    predictions = evaluate_model(model)

# Add predictions to DataFrame and save
predictions = np.array(predictions).squeeze()
data[['predicted-remember', 'predicted-understand', 'predicted-apply', 'predicted-analyze', 'predicted-evaluate', 'predicted-create']] = predictions
data.to_csv('intern chatbot - myquiz_with_predictions.csv', index=False)
