#!/usr/bin/env bash

isort . -s venv -s web && black . --exclude='/(venv|web)/' --line-length=99 --experimental-string-processing