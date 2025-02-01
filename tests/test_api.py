from re import A
from typing import Any
import pytest
from fastapi.testclient import TestClient
from auto_post_classifier.models import Post

# Import your FastAPI app here - you'll need to create/expose it
from main import app

client = TestClient(app)

"""
What is being tested:
- The rank endpoint with no posts
- The rank endpoint with invalid posts
- The rank endpoint with valid posts
"""
# TODO: in dict[str, Any] -> replace Any with api.Post: this causes a "json object not serializable" error

@pytest.fixture
def empty_post() -> dict[str, Any]:
    return {"text": "", "content_url": ""}


@pytest.fixture
def invalid_posts() -> dict[str, Any]:
    return {"some_key": {"unrelated_key": "unrelated_value"}}


@pytest.fixture
def valid_posts() -> dict[str, Any]:
    return {"test_id": {"text": "Hello World", "content_url": "https://example.com"}}


def test_rank_endpoint_with_no_posts():
    """Test the rank endpoint with no posts"""
    # Send POST request to the rank endpoint
    response = client.post("/rank", json={})

    # Assert response status code is 200 (OK)
    assert response.status_code == 200


def test_rank_endpoint_with_invalid_posts(invalid_posts):
    """Test the rank endpoint with invalid posts"""
    # Prepare test data
    # Send POST request to the rank endpoint
    response = client.post("/rank", json=invalid_posts)

    # Assert response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422


def test_rank_endpoint_with_valid_posts(valid_posts):
    """Test the rank endpoint with a simple hello world example"""
    # Prepare test data

    # Send POST request to the rank endpoint
    response = client.post("/rank", json=valid_posts)

    # Assert response status code is 200 (OK)
    assert response.status_code == 200


def test_rank_endpoint_fails_with_post_empty_text(empty_post):
    """Test the rank endpoint with a post with empty text"""
    # Prepare test data
    # Send POST request to the rank endpoint
    response = client.post("/rank", json=empty_post)

    # Assert response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422
