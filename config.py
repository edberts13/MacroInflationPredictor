import os

# ── FRED API Key ──────────────────────────────────────────────
# Production (Railway): set FRED_API_KEY as an environment variable
#   in Railway Dashboard → Variables → FRED_API_KEY = your_key
# Local dev: falls back to the hardcoded key below if env var not set.
# Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

FRED_API_KEY = os.environ.get("FRED_API_KEY", "8ac115c5e35efd3ce066f2f30c840a64")
