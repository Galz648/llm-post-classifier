import pytest
from fastapi.testclient import TestClient
from auto_post_classifier.models import Post

# Import your FastAPI app here - you'll need to create/expose it
from main import Posts, app

client = TestClient(app)

"""
What is being tested:
- The rank endpoint with no posts
- The rank endpoint with invalid posts
- The rank endpoint with valid posts
"""
@pytest.fixture
def valid_posts() -> Posts:
    return Posts(data={"test_id": Post(text="Hello World", content_url="https://example.com")})


def test_rank_endpoint_with_no_posts():
    """Test the rank endpoint with no posts"""
    # Prepare test data
    posts = Posts(data={})
    # Send POST request to the rank endpoint
    response = client.post("/rank", json=posts)

    # Assert response status code is 200 (OK)
    assert response.status_code == 200



def test_rank_endpoint_with_invalid_posts():
    """Test the rank endpoint with invalid posts"""
    # Prepare test data
    posts = Posts(data={"test_id": Post(text="Hello World", content_url="https://example.com")})
    # Send POST request to the rank endpoint
    response = client.post("/rank", json=posts)

    # Assert response status code is 200 (OK)
    assert response.status_code == 200


def test_rank_endpoint_with_valid_posts(valid_posts):
    """Test the rank endpoint with a simple hello world example"""
    # Prepare test data
    posts = Posts(data=valid_posts)
    # Send POST request to the rank endpoint
    response = client.post("/rank", json=posts)

    # Assert response status code is 200 (OK)
    assert response.status_code == 200

    # Assert response contains the expected structure
    response_data = response.json()
    assert "test_id" in response_data

    # The response should contain classification results
    result = response_data["test_id"]
    assert isinstance(result, dict)
    # Add more specific assertions based on your expected response structure
