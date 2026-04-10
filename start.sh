#!/usr/bin/env sh
set -e

if [ "${RUN_MODE:-api}" = "inference" ]; then
	python inference.py
else
	uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}"
fi
