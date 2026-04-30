#!/bin/bash
# manifestme.sh
# Write manifest.json (+ bundle metadata) for Posit Connect deployment of this Shiny for Python app.
# Entry point is auto-detected (Shiny Express → e.g. shiny.express.app:app_*_drug_*).
#
# Run from this folder:
#   chmod +x manifestme.sh && ./manifestme.sh
# Or from anywhere:
#   bash /path/to/tool2/app/manifestme.sh
#
# Tip: use a virtualenv with only `pip install -r requirements.txt` so the captured
# environment stays small. Otherwise rsconnect may freeze your whole conda base.

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

pip install -q rsconnect-python
rsconnect write-manifest shiny --overwrite .
