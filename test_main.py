import json
import os

import pandas as pd
from fastapi.testclient import TestClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import main
from auto_post_classifier.gpt_handler import GPT_ERROR_REASONS

client = TestClient(main.app)
sample_name = "sample_1.json"
sample_path = f"tests/samples/{sample_name}"
# os.environ["MOCK_FILE"] = f"mock/{sample_name}"


def set_up_tests():
    with open(sample_path) as f:
        data = json.load(f)

    request = data["request"]
    number_of_lines_in_sample = len(request)
    volunteers_validation = data["wanted_responses"]

    response = client.post("/rank", json=request)
    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data, dict)

    for uuid, response_entry in response_data.items():
        assert uuid in volunteers_validation, "wrong mock file for the test"
        response_entry["volunteers"] = volunteers_validation[uuid]

    df = pd.DataFrame.from_dict(response_data, orient="index")

    # fixme: fix this
    df.to_csv("test.csv")
    df = pd.read_csv("test.csv")

    return df, number_of_lines_in_sample


df, number_of_lines_in_sample = set_up_tests()


def test_validation():
    assert df.loc[:, "error"].isnull().all()


def test_response_have_all_keys():
    assert len(df) == number_of_lines_in_sample


def test_many_requests_error():
    assert GPT_ERROR_REASONS.TO_MANY_REQUESTS not in df["error"].unique()


def test_json_validation_error():
    assert GPT_ERROR_REASONS.JSON_VALIDARTION not in df["error"].unique()


def test_classification_performance():
    """Test the performance of the classifier against wanted responses"""
    
    # Get actual and predicted labels
    actual_labels = df['volunteers'].tolist()
    predicted_labels = df['category'].tolist()
    
    # Get unique labels
    unique_labels = sorted(list(set(actual_labels + predicted_labels)))
    
    # Calculate confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=unique_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels,
                yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Calculate and print classification report
    report = classification_report(actual_labels, predicted_labels)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(actual_labels) == np.array(predicted_labels))
    
    # Plot accuracy per category
    category_accuracies = {}
    for category in unique_labels:
        mask = np.array(actual_labels) == category
        if mask.sum() > 0:
            cat_accuracy = np.mean(np.array(predicted_labels)[mask] == category)
            category_accuracies[category] = cat_accuracy
    
    plt.figure(figsize=(10, 6))
    plt.bar(category_accuracies.keys(), category_accuracies.values())
    plt.title('Accuracy per Category')
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('category_accuracies.png')
    plt.close()
    
    # Assert minimum accuracy threshold
    min_accuracy_threshold = 0.7  # You can adjust this threshold
    assert accuracy >= min_accuracy_threshold, f"Overall accuracy {accuracy:.2f} is below minimum threshold {min_accuracy_threshold}"


def test_response_distribution():
    """Test and visualize the distribution of responses"""
    
    # Plot distribution of actual vs predicted labels
    plt.figure(figsize=(12, 6))
    
    # Create subplots
    plt.subplot(1, 2, 1)
    df['volunteers'].value_counts().plot(kind='bar')
    plt.title('Distribution of Actual Labels')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    df['category'].value_counts().plot(kind='bar')
    plt.title('Distribution of Predicted Labels')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('label_distributions.png')
    plt.close()
    
    # Assert that we have predictions for all categories
    assert set(df['volunteers'].unique()) == set(df['category'].unique()), \
        "Mismatch between actual and predicted category sets"


def test_platform_performance():
    """Test classifier performance across different platforms"""
    
    platforms = df['platform'].unique()
    platform_accuracies = {}
    
    for platform in platforms:
        platform_df = df[df['platform'] == platform]
        accuracy = np.mean(platform_df['volunteers'] == platform_df['category'])
        platform_accuracies[platform] = accuracy
    
    # Plot platform accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(platform_accuracies.keys(), platform_accuracies.values())
    plt.title('Accuracy by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('platform_accuracies.png')
    plt.close()
    
    # Assert minimum accuracy per platform
    min_platform_accuracy = 0.6  # You can adjust this threshold
    for platform, accuracy in platform_accuracies.items():
        assert accuracy >= min_platform_accuracy, \
            f"Accuracy for {platform} ({accuracy:.2f}) is below minimum threshold {min_platform_accuracy}"
