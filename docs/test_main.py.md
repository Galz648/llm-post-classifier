# Test Main Documentation

## Overview
This test suite evaluates the performance and functionality of an auto post classifier system. It performs various tests including validation checks, error handling, classification performance analysis, and platform-specific performance evaluation.

## Dependencies
- `json`: For JSON file handling
- `pandas`: For data manipulation and analysis
- `fastapi.testclient`: For API testing
- `matplotlib` & `seaborn`: For visualization
- `sklearn.metrics`: For classification metrics
- `numpy`: For numerical operations

## Global Setup
```python
client = TestClient(main.app)
sample_name = "sample_1.json"
sample_path = f"tests/samples/{sample_name}"
```
The test suite uses a FastAPI test client and loads test data from a sample JSON file located in `tests/samples/`.

## Core Functions

### `set_up_tests()`
Sets up the test environment and prepares the data for testing.

**Process:**
1. Loads sample data from JSON file
2. Makes a POST request to "/rank" endpoint
3. Validates response format
4. Merges response with volunteer validation data
5. Creates a pandas DataFrame for analysis

**Returns:**
- `df`: DataFrame containing test results
- `number_of_lines_in_sample`: Number of samples in test data

### `test_validation()`
Verifies that no errors occurred during processing.

**Assertion:**
- All entries in the 'error' column should be null

### `test_response_have_all_keys()`
Ensures the response contains all expected samples.

**Assertion:**
- Number of responses matches number of input samples

### `test_many_requests_error()`
Checks for rate limiting errors.

**Assertion:**
- No "TO_MANY_REQUESTS" errors in response

### `test_json_validation_error()`
Validates JSON formatting.

**Assertion:**
- No JSON validation errors in response

### `test_classification_performance()`
Comprehensive evaluation of classifier performance.

**Components:**
1. **Confusion Matrix**
   - Visualizes true vs predicted labels
   - Saved as 'confusion_matrix.png'

2. **Classification Report**
   - Generates detailed metrics (precision, recall, f1-score)
   - Saved as 'classification_report.txt'

3. **Category-wise Accuracy**
   - Calculates accuracy for each category
   - Visualizes results in bar chart
   - Saved as 'category_accuracies.png'

**Assertion:**
- Overall accuracy must exceed minimum threshold (default: 0.7)

### `test_response_distribution()`
Analyzes distribution of predictions vs actual labels.

**Visualizations:**
1. Distribution of actual labels (volunteers)
2. Distribution of predicted labels
- Combined plot saved as 'label_distributions.png'

**Assertion:**
- Predicted categories must match set of actual categories

### `test_platform_performance()`
Evaluates classifier performance across different platforms.

**Process:**
1. Calculates accuracy per platform
2. Generates visualization of platform-wise performance
   - Saved as 'platform_accuracies.png'

**Assertion:**
- Each platform must meet minimum accuracy threshold (default: 0.6)

## Output Files
The test suite generates several visualization files:
1. `confusion_matrix.png`: Confusion matrix visualization
2. `classification_report.txt`: Detailed classification metrics
3. `category_accuracies.png`: Per-category accuracy visualization
4. `label_distributions.png`: Distribution comparison visualization
5. `platform_accuracies.png`: Platform-wise performance visualization
6. `test.csv`: Intermediate test results

## Usage
Run the tests using pytest:
```bash
pytest test_main.py
```

## Notes
- The test suite requires a properly formatted sample JSON file in the tests/samples directory
- Visualization files are saved in the working directory
- Minimum accuracy thresholds can be adjusted in the code:
  - Overall accuracy threshold: 0.7 (in test_classification_performance)
  - Platform accuracy threshold: 0.6 (in test_platform_performance)

This documentation provides a comprehensive overview of the test suite's functionality, requirements, and outputs. It can be placed in the same directory as the test file or in a dedicated documentation folder.

Would you like me to expand on any particular section or add additional details?
