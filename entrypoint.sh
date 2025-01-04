#!/bin/bash
echo running script
# poetry run python auto_post_classifier/main.py -d data/AntiIsraeli.csv --no-shuffle --no-api -n -1
poetry run uvicorn main:app --host 0.0.0.0 --port 80
