import pandas as pd
import torch
import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('intern chatbot - myquiz.csv')

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# Shortened skill names for plotting
skill_names = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

# Function to classify a question using the local Gemma model
def classify_question(question):
    url = "http://localhost:8080/api/generate"
    payload = {
        "model": "gemma:2b",
        "prompt": question,
        "max_new_tokens": 256
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses
        result = response.json()
        generated_text = result.get('data', [{}])[0].get('text', '')

        # Print the generated text for debugging
        print("Generated text:", generated_text)

        classification = {skill: 1.0 if skill in generated_text else 0.0 for skill in skill_names}
        return classification

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {skill: 0.0 for skill in skill_names}

# Function to evaluate the model
def evaluate_model():
    total_loss = 0
    all_true_labels = []
    all_predicted_labels = []

    for i, question in enumerate(data['question']):
        classification = classify_question(question)
        pred = [classification.get(skill, 0) for skill in skill_names]

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
    accuracy = (all_true_labels == all_predicted_labels).mean()

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
    plt.savefig('confusion_matrix.png')
    plt.show()

# Load ground truth labels
true_labels = data[['ind-remember', 'ind-understand', 'ind-apply', 'ind-analyze', 'ind-evaluate', 'ind-create']].values

# Evaluate the model
evaluate_model()


# # Save predictions to CSV
# predictions = np.vstack(all_predicted_labels)
# pred_df = pd.DataFrame(predictions, columns=['predicted-remember', 'predicted-understand', 'predicted-apply', 'predicted-analyze', 'predicted-evaluate', 'predicted-create'])
# data = pd.concat([data, pred_df], axis=1)
# data.to_csv('intern_chatbot_with_predictions.csv', index=False)




# have some context
# sci math etc etc
# want to try. Put in the entire content, and use it to generate questions
# multiple choice


# variable difficulty?
# for blooms taxonomy
# number of questions