#!/bin/bash
# deployme.sh
# Deploy this Shiny for Python app to Posit Connect via rsconnect-python.
#
# Prerequisites: .env in this folder with CONNECT_SERVER and CONNECT_API_KEY (see .env.example).
#
# Run from this folder:
#   chmod +x deployme.sh && ./deployme.sh
# Or from anywhere:
#   bash /path/to/tool2/app/deployme.sh
#
# App entry: Shiny Express in app_drug.py (do not pass -e unless you rename the file).

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -f .env ]; then
  echo "deployme.sh: missing .env in $DIR" >&2
  echo "Copy .env.example to .env and set CONNECT_SERVER and CONNECT_API_KEY." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
# Strip CR so Windows-style line endings do not break variable names (CONNECT_SERVER^M).
source /dev/stdin <<< "$(tr -d '\r' < .env)"
set +a

# Trim leading/trailing whitespace (common .env typo); rsconnect raises "URL does not contain a hostname" if blank or malformed.
CONNECT_SERVER="$(printf '%s' "${CONNECT_SERVER:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
CONNECT_API_KEY="$(printf '%s' "${CONNECT_API_KEY:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
export CONNECT_SERVER CONNECT_API_KEY

if [ -z "${CONNECT_SERVER:-}" ]; then
  echo "deployme.sh: CONNECT_SERVER is empty or missing in $DIR/.env" >&2
  echo "Add a line like: CONNECT_SERVER=https://connect.posit.cloud" >&2
  echo "(no # at the start; no spaces around =)" >&2
  exit 1
fi

# Fix a common typo: CONNECT_SERVER=https:https://host/path
if [[ "$CONNECT_SERVER" == https:https://* ]]; then
  echo "deployme.sh: detected CONNECT_SERVER starting with 'https:https://'; fixing to 'https://...'" >&2
  CONNECT_SERVER="${CONNECT_SERVER#https:}"
fi
if [[ "$CONNECT_SERVER" == http:http://* ]]; then
  echo "deployme.sh: detected CONNECT_SERVER starting with 'http:http://'; fixing to 'http://...'" >&2
  CONNECT_SERVER="${CONNECT_SERVER#http:}"
fi

# Accept bare hostnames (e.g. connect.posit.cloud) — common .env mistake; rsconnect needs a full URL.
case "$CONNECT_SERVER" in
  http://*|https://*) ;;
  *)
    if [[ "$CONNECT_SERVER" == *.* ]] && [[ "$CONNECT_SERVER" != *" "* ]]; then
      echo "deployme.sh: note: prepending https:// to CONNECT_SERVER (set full URL in .env to silence this)." >&2
      CONNECT_SERVER="https://${CONNECT_SERVER}"
    else
      echo "deployme.sh: CONNECT_SERVER must be a URL or hostname (with dots), e.g.:" >&2
      echo "  CONNECT_SERVER=https://connect.posit.cloud" >&2
      echo "  CONNECT_SERVER=connect.posit.cloud" >&2
      exit 1
    fi
    ;;
esac

_connect_rest="$CONNECT_SERVER"
case "$_connect_rest" in
  https://*) _connect_rest="${_connect_rest#https://}" ;;
  http://*) _connect_rest="${_connect_rest#http://}" ;;
esac
CONNECT_HOST="${_connect_rest%%/*}"
if [ -z "$CONNECT_HOST" ]; then
  echo "deployme.sh: CONNECT_SERVER has no hostname after https:// (rsconnect: \"URL does not contain a hostname\")." >&2
  echo "Use: CONNECT_SERVER=https://connect.posit.cloud" >&2
  echo "Not: CONNECT_SERVER=https:// or only a path — check $DIR/.env" >&2
  exit 1
fi
unset _connect_rest CONNECT_HOST
export CONNECT_SERVER

# Prefer .python-version (rsconnect deprecated --override-python-version).
PYVER="${CONNECT_PYTHON_VERSION:-3.12.4}"
if [ ! -f .python-version ]; then
  printf '%s\n' "$PYVER" > .python-version
  echo "deployme.sh: wrote .python-version=$PYVER" >&2
fi

if [ -z "${CONNECT_API_KEY:-}" ]; then
  echo "deployme.sh: CONNECT_API_KEY is empty or missing in $DIR/.env" >&2
  echo "Create a key in Posit Connect: profile → API keys → New API key." >&2
  exit 1
fi

pip install -q rsconnect-python

TITLE="${CONNECT_TITLE:-drugsfda-explorer}"

rsconnect deploy shiny \
  --title "$TITLE" \
  --server "$CONNECT_SERVER" \
  --api-key "$CONNECT_API_KEY" \
  -x ".env" \
  -x ".git" \
  -x ".gitignore" \
  -x "*.pyc" \
  -x "**/__pycache__" \
  .
