"""
Auto Post Classifier Test Suite
==============================

This module contains comprehensive tests for the auto post classifier system.
It evaluates classification performance, error handling, and generates 
visualization reports.

The test suite requires:
- A properly configured FastAPI application
- Sample data in JSON format
- Access to the classifier API endpoint
"""

import json
import os

import pandas as pd
from fastapi.testclient import TestClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from auto_post_classifier.gpt_handler import GPT_ERROR_REASONS
from utils import TestReportGenerator

import main
import pytest

# Initialize test client and set up test data paths
# TODO: remove constants from the test file
client = TestClient(main.app)
sample_name = "sample_1.json"
sample_path = f"tests/samples/{sample_name}"
num_elements = 1


def load_sample_data():
    with open(sample_path) as f:
        data = json.load(f)
    return data


# Fixture to return the first elements of the list
@pytest.fixture
def get_elements():
    data = load_sample_data()
    return data[: min(num_elements, len(data))]


def set_up_tests():
    """
    Prepares the test environment and loads test data.

    This function:
    1. Loads sample posts from JSON
    2. Sends posts to classifier API
    3. Validates responses
    4. Merges with volunteer validation data
    5. Creates a DataFrame for analysis

    Returns:
        tuple: (DataFrame with test results, number of samples tested)

    Raises:
        AssertionError: If response format is invalid or mock file mismatch
    """
    with open(sample_path) as f:
        data = json.load(f)

    request = data["request"]
    number_of_lines_in_sample = len(request)
    volunteers_validation = data["wanted_responses"]
    platforms = data["platforms"]

    # Send request to classifier API
    response = client.post("/rank", json=request)
    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data, dict)

    # Merge classifier responses with volunteer validations
    for uuid, response_entry in response_data.items():
        assert uuid in volunteers_validation, "wrong mock file for the test"
        response_entry["volunteers"] = volunteers_validation[uuid]
        response_entry["platform"] = platforms[uuid]

    # Create DataFrame for analysis
    df = pd.DataFrame.from_dict(response_data, orient="index")

    # Generate reports
    report_generator = TestReportGenerator()

    # Generate HTML report
    html_path = "test_reports/classification_report.html"
    report_generator.generate_report(df, html_path)

    # Generate PDF report with visualizations
    pdf_path = "test_reports/classification_report.pdf"
    report_generator.generate_pdf_report(df, pdf_path)

    # TODO: Remove temporary CSV creation step
    df.to_csv("test.csv")
    df = pd.read_csv("test.csv")

    return df, number_of_lines_in_sample


# Initialize test data
df, number_of_lines_in_sample = set_up_tests()


def test_validation():
    """Verify that no processing errors occurred during classification."""
    assert df.loc[:, "error"].isnull().all()


def test_response_have_all_keys():
    """Ensure all input samples received a classification response."""
    assert len(df) == number_of_lines_in_sample


def test_many_requests_error():
    """Check that no rate limiting errors occurred."""
    assert GPT_ERROR_REASONS.TO_MANY_REQUESTS not in df["error"].unique()


def test_classification_performance():
    """
    Evaluate classifier performance against volunteer validations.

    Generates:
        - Confusion matrix visualization
        - Classification metrics report
        - Category-wise accuracy plots

    Asserts:
        - Overall accuracy meets minimum threshold (0.7)
    """
    # Extract actual and predicted labels
    actual_labels = df["volunteers"].tolist()
    predicted_labels = df["category"].tolist()
    unique_labels = sorted(list(set(actual_labels + predicted_labels)))

    # Generate and save confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Generate and save classification report
    report = classification_report(actual_labels, predicted_labels)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Calculate overall accuracy
    accuracy = np.mean(np.array(actual_labels) == np.array(predicted_labels))

    # Calculate and plot category-wise accuracies
    category_accuracies = {}
    for category in unique_labels:
        mask = np.array(actual_labels) == category
        if mask.sum() > 0:
            cat_accuracy = np.mean(np.array(predicted_labels)[mask] == category)
            category_accuracies[category] = cat_accuracy

    plt.figure(figsize=(10, 6))
    plt.bar(category_accuracies.keys(), category_accuracies.values())
    plt.title("Accuracy per Category")
    plt.xlabel("Category")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("category_accuracies.png")
    plt.close()

    # Verify minimum accuracy threshold
    min_accuracy_threshold = 0.7
    assert (
        accuracy >= min_accuracy_threshold
    ), f"Overall accuracy {accuracy:.2f} is below minimum threshold {min_accuracy_threshold}"


def test_response_distribution():
    """
    Analyze and visualize the distribution of predictions vs actual labels.

    Generates:
        - Side-by-side bar plots of actual vs predicted label distributions

    Asserts:
        - Predicted categories match set of actual categories
    """
    plt.figure(figsize=(12, 6))

    # Plot actual label distribution
    plt.subplot(1, 2, 1)
    df["volunteers"].value_counts().plot(kind="bar")
    plt.title("Distribution of Actual Labels")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Plot predicted label distribution
    plt.subplot(1, 2, 2)
    df["category"].value_counts().plot(kind="bar")
    plt.title("Distribution of Predicted Labels")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("label_distributions.png")
    plt.close()

    # Verify category consistency
    assert set(df["volunteers"].unique()) == set(
        df["category"].unique()
    ), "Mismatch between actual and predicted category sets"


def test_platform_performance():
    """
    Evaluate classifier performance across different platforms.

    Generates:
        - Bar plot of accuracy by platform

    Asserts:
        - Each platform meets minimum accuracy threshold (0.6)
    """
    platforms = df["platform"].unique()
    platform_accuracies = {}

    # Calculate accuracy for each platform
    for platform in platforms:
        platform_df = df[df["platform"] == platform]
        accuracy = np.mean(platform_df["volunteers"] == platform_df["category"])
        platform_accuracies[platform] = accuracy

    # Visualize platform accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(platform_accuracies.keys(), platform_accuracies.values())
    plt.title("Accuracy by Platform")
    plt.xlabel("Platform")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("platform_accuracies.png")
    plt.close()

    # Verify platform-specific accuracy thresholds
    min_platform_accuracy = 0.6
    for platform, accuracy in platform_accuracies.items():
        assert (
            accuracy >= min_platform_accuracy
        ), f"Accuracy for {platform} ({accuracy:.2f}) is below minimum threshold {min_platform_accuracy}"
