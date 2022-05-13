#!/bin/bash

pip install --upgrade pip # sometimes required to avoid OOM
pip install .

# This runs your wrapped command
"$@"
