# import auto_post_classifier.gpt_handler as gpt_handler
from auto_post_classifier.request_builder import RequestBuilder
import pytest


def test_request_throws_if_config_is_not_provided():
    builder = RequestBuilder()

    with pytest.raises(ValueError):
        builder.build()


def test_request_builder_throws_if_text_is_not_provided():
    builder = RequestBuilder()

    with pytest.raises(ValueError):
        builder.add_text_support("")


def test_request_builder_adds_text_support():
    builder = RequestBuilder()
    builder.add_text_support("test")

    assert builder.config.messages[0].content[0].text == "test"
