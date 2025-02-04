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
from auto_post_classifier.models import Post
import pandas as pd
from fastapi.testclient import TestClient
import matplotlib.pyplot as plt

from auto_post_classifier.gpt_handler import GPT_ERROR_REASONS
from utils import TestReportGenerator

import main
import pytest

# Constants
SAMPLE_NAME = "sample_1.json"
SAMPLE_PATH = f"tests/samples/{SAMPLE_NAME}"
NUM_ELEMENTS = 1
HTML_REPORT_PATH = "test_reports/classification_report.html"
PDF_REPORT_PATH = "test_reports/classification_report.pdf"
TEMP_CSV_PATH = "test.csv"

# TODO
# Initialize test client and set up test data paths
# TODO: remove constants from the test file
client = TestClient(main.app)


@pytest.fixture(autouse=True)
def run_before_each_test():
    """
    Fixture to run before each test.
    """
    # Load sample data
    data = load_sample_data()

    # Merge responses with validation data
    merged_data = merge_responses(
        data["request"], data["wanted_responses"], data["platforms"]
    )

    # Create DataFrame and generate reports
    df = pd.DataFrame.from_dict(merged_data, orient="index")
    generate_test_reports(df, HTML_REPORT_PATH, PDF_REPORT_PATH)

    # TODO: Remove temporary CSV creation step
    df.to_csv(TEMP_CSV_PATH)
    df = pd.read_csv(TEMP_CSV_PATH)

    # Set global variables for tests
    global test_df, number_of_lines_in_sample
    test_df = df
    number_of_lines_in_sample = len(data["request"])


def load_sample_data():
    try:
        with open(SAMPLE_PATH) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading sample data: {e}")
        raise e
    return data


# Fixture to return the first elements of the list
# TODO: change fixture name
@pytest.fixture
def get_elements():
    data = load_sample_data()
    try:
        validated_data = [Post(**item).model_dump() for item in data]
    except Exception as e:
        print(f"Validation error: {e}")
    return validated_data[: min(NUM_ELEMENTS, len(data))]


def load_test_data(sample_path: str) -> tuple[list, dict, dict]:
    """
    Load test data from the sample JSON file.

    Args:
        sample_path: Path to the sample JSON file

    Returns:
        tuple: (request data, volunteer validations, platforms)
    """
    with open(sample_path) as f:
        data = json.load(f)

    # validate the model
    data = Post.model_validate_json(data)

    return data


def make_classification_request(request_data: list, endpoint: str = "/rank") -> dict:
    """
    Send request to classifier API and validate response.

    Args:
        request_data: List of items to classify
        endpoint: API endpoint to call

    Returns:
        dict: API response data

    Raises:
        AssertionError: If response format is invalid
    """
    response = client.post(endpoint, json=request_data)
    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data, dict)
    return response_data


#
def merge_responses(
    response_data: dict, volunteers_validation: dict, platforms: dict
) -> dict:
    """
    Merge classifier responses with volunteer validations and platform data.

    Args:
        response_data: Classifier API responses
        volunteers_validation: Ground truth from volunteers
        platforms: Platform information for each item

    Returns:
        dict: Merged data

    Raises:
        AssertionError: If response UUID not found in validation data
    """
    merged_data = {}
    for uuid, response_entry in response_data.items():
        assert uuid in volunteers_validation, "wrong mock file for the test"
        merged_entry = response_entry.copy()
        merged_entry["volunteers"] = volunteers_validation[uuid]
        merged_entry["platform"] = platforms[uuid]
        merged_data[uuid] = merged_entry
    return merged_data


def generate_test_reports(df: pd.DataFrame, html_path: str, pdf_path: str) -> None:
    """
    Generate HTML and PDF reports from test results.

    Args:
        df: DataFrame containing test results
        html_path: Path where HTML report should be saved
        pdf_path: Path where PDF report should be saved
    """
    report_generator = TestReportGenerator()

    # Generate HTML report
    report_generator.generate_report(df, html_path)

    # Generate PDF report with visualizations
    report_generator.generate_pdf_report(df, pdf_path)


def set_up_tests() -> tuple[pd.DataFrame, int]:
    """
    Prepares the test environment and loads test data.

    This function coordinates the test setup process by:
    1. Loading sample posts from JSON
    2. Sending posts to classifier API
    3. Validating responses
    4. Merging with volunteer validation data
    5. Creating a DataFrame for analysis

    Returns:
        tuple: (DataFrame with test results, number of samples tested)
    """
    # Load test data
    request_data, volunteers_validation, platforms = load_test_data(SAMPLE_PATH)
    number_of_lines_in_sample = len(request_data)

    # Get and validate classifier responses
    response_data = make_classification_request(request_data)

    # Merge responses with validation data
    merged_data = merge_responses(response_data, volunteers_validation, platforms)

    # Create DataFrame and generate reports
    df = pd.DataFrame.from_dict(merged_data, orient="index")
    generate_test_reports(df, HTML_REPORT_PATH, PDF_REPORT_PATH)

    # TODO: Remove temporary CSV creation step
    df.to_csv(TEMP_CSV_PATH)
    df = pd.read_csv(TEMP_CSV_PATH)

    return df, number_of_lines_in_sample


# Initialize test data
# df, number_of_lines_in_sample = set_up_tests()


# def test_validation():
#     """Verify that no processing errors occurred during classification."""
#     assert df.loc[:, "error"].isnull().all()


# def test_response_have_all_keys():
#     """Ensure all input samples received a classification response."""
#     assert len(df) == number_of_lines_in_sample


# def test_many_requests_error():
#     """Check that no rate limiting errors occurred."""
#     assert GPT_ERROR_REASONS.TO_MANY_REQUESTS not in df["error"].unique()


# def test_classification_performance():
#     """
#     Evaluate classifier performance against volunteer validations.

#     Generates:
#         - Confusion matrix visualization
#         - Classification metrics report
#         - Category-wise accuracy plots

#     Asserts:
#         - Overall accuracy meets minimum threshold (0.7)
#     """
#     # Extract actual and predicted labels
#     actual_labels = df["volunteers"].tolist()
#     predicted_labels = df["category"].tolist()
#     unique_labels = sorted(list(set(actual_labels + predicted_labels)))

#     # Generate and save confusion matrix
#     cm = confusion_matrix(actual_labels, predicted_labels, labels=unique_labels)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         xticklabels=unique_labels,
#         yticklabels=unique_labels,
#     )
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("confusion_matrix.png")
#     plt.close()

#     # Generate and save classification report
#     report = classification_report(actual_labels, predicted_labels)
#     with open("classification_report.txt", "w") as f:
#         f.write(report)

#     # Calculate overall accuracy
#     accuracy = np.mean(np.array(actual_labels) == np.array(predicted_labels))

#     # Calculate and plot category-wise accuracies
#     category_accuracies = {}
#     for category in unique_labels:
#         mask = np.array(actual_labels) == category
#         if mask.sum() > 0:
#             cat_accuracy = np.mean(np.array(predicted_labels)[mask] == category)
#             category_accuracies[category] = cat_accuracy

#     plt.figure(figsize=(10, 6))
#     plt.bar(category_accuracies.keys(), category_accuracies.values())
#     plt.title("Accuracy per Category")
#     plt.xlabel("Category")
#     plt.ylabel("Accuracy")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("category_accuracies.png")
#     plt.close()

#     # Verify minimum accuracy threshold
#     min_accuracy_threshold = 0.7
#     assert (
#         accuracy >= min_accuracy_threshold
#     ), f"Overall accuracy {accuracy:.2f} is below minimum threshold {min_accuracy_threshold}"


# def test_response_distribution():
#     """
#     Analyze and visualize the distribution of predictions vs actual labels.

#     Generates:
#         - Side-by-side bar plots of actual vs predicted label distributions

#     Asserts:
#         - Predicted categories match set of actual categories
#     """
#     plt.figure(figsize=(12, 6))

#     # Plot actual label distribution
#     plt.subplot(1, 2, 1)
#     df["volunteers"].value_counts().plot(kind="bar")
#     plt.title("Distribution of Actual Labels")
#     plt.xlabel("Category")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45)

#     # Plot predicted label distribution
#     plt.subplot(1, 2, 2)
#     df["category"].value_counts().plot(kind="bar")
#     plt.title("Distribution of Predicted Labels")
#     plt.xlabel("Category")
#     plt.ylabel("Count")
#     plt.xticks(rotation=45)

#     plt.tight_layout()
#     plt.savefig("label_distributions.png")
#     plt.close()

#     # Verify category consistency
#     assert set(df["volunteers"].unique()) == set(
#         df["category"].unique()
#     ), "Mismatch between actual and predicted category sets"


# def test_platform_performance():
#     """
#     Evaluate classifier performance across different platforms.

#     Generates:
#         - Bar plot of accuracy by platform

#     Asserts:
#         - Each platform meets minimum accuracy threshold (0.6)
#     """
#     platforms = df["platform"].unique()
#     platform_accuracies = {}

#     # Calculate accuracy for each platform
#     for platform in platforms:
#         platform_df = df[df["platform"] == platform]
#         accuracy = np.mean(platform_df["volunteers"] == platform_df["category"])
#         platform_accuracies[platform] = accuracy

#     # Visualize platform accuracies
#     plt.figure(figsize=(10, 6))
#     plt.bar(platform_accuracies.keys(), platform_accuracies.values())
#     plt.title("Accuracy by Platform")
#     plt.xlabel("Platform")
#     plt.ylabel("Accuracy")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("platform_accuracies.png")
#     plt.close()

#     # Verify platform-specific accuracy thresholds
#     min_platform_accuracy = 0.6
#     for platform, accuracy in platform_accuracies.items():
#         assert (
#             accuracy >= min_platform_accuracy
#         ), f"Accuracy for {platform} ({accuracy:.2f}) is below minimum threshold {min_platform_accuracy}"
