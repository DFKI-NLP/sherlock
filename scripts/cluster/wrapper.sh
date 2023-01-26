#!/bin/bash

pip install --upgrade pip 2>&1 # sometimes required to avoid OOM
pip install . 2>&1
pip install pyOpenSSL --upgrade  # https://askubuntu.com/questions/1428181/module-lib-has-no-attribute-x509-v-flag-cb-issuer-check
# use this to solve SHM error? https://chat.opendfki.de/km-all/pl/6i1ffd8zb7friponj11icfkxkw
ulimit -n 64000

# This runs your wrapped command
"$@"
