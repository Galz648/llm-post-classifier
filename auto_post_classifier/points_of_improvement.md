# Points of Improvement
* separate the tests into metric generation and figure generation
* use pydantic to validate the response from the API (response_validator (gpt_handler.py))
* export constants into a separate file
* add a linter - Makefile
* add a pre-commit hook - maybe Makefile?
* create a separation in tests between the metrics and the figures, as well as unit tests and integration tests (testing the whole external api call)

