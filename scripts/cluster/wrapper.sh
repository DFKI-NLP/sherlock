#!/bin/bash

pip install --upgrade pip 2>&1 # sometimes required to avoid OOM
pip install . 2>&1

# This runs your wrapped command
"$@"
