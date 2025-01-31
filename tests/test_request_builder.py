# import auto_post_classifier.gpt_handler as gpt_handler
from auto_post_classifier.request_builder import RequestBuilder
import pytest


def test_request_throws_if_config_is_not_provided_on_build():
    builder = RequestBuilder()

    with pytest.raises(ValueError):
        builder.build()


def test_request_builder_adds_text_support():
    builder = RequestBuilder()
    builder.add_text_support("test")

    assert builder.config.messages.pop().content.pop().text == "test"
    assert len(builder.config.messages) == 1 # the system prompt message is added by default; TODO: determine if this is coupled to the implementation


