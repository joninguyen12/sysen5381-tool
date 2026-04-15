# Load .env from this directory (where app_drug.py lives), not only the process cwd.
# Order: `.env` first, then `.env.txt` (fills any vars still unset; rename .env.txt → .env when ready).
from pathlib import Path

from dotenv import load_dotenv

_dir = Path(__file__).resolve().parent
load_dotenv(_dir / ".env")
load_dotenv(_dir / ".env.txt")
