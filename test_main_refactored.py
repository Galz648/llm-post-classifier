import pytest
import json
from pydantic import BaseModel, ValidationError

# Constants
SAMPLE_NAME = "sample_1.json"
SAMPLE_PATH = f"tests/samples/{SAMPLE_NAME}"

class RequestModel(BaseModel):
    request: dict
    wanted_responses: dict
    platforms: dict

class TestAutoPostClassifier():
    
    @pytest.fixture(scope="class")
    def test_validate_data(self):
        with open(SAMPLE_PATH) as f:
            data = json.load(f)
        
        try:
            validated_data = RequestModel.model_validate(data)
        except ValidationError as e:
            pytest.fail(f"Validation error: {e}")
        
        return validated_data

    # def test_data_format(self, setup_class):
    #     """Test that the data format is valid according to the RequestModel."""
    #     assert isinstance(setup_class, RequestModel)
    
    # def test_request_keys(self, setup_class):
    #     """Test that the request contains the expected keys."""
    #     assert "request" in setup_class.dict()
    #     assert "wanted_responses" in setup_class.dict()
    #     assert "platforms" in setup_class.dict()


 