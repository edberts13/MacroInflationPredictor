# ─────────────────────────────────────────────────────────────────────────────
# Macro Inflation Predictor — Production Dockerfile
#
# PURPOSE: Serve the Streamlit dashboard ONLY.
#          Does NOT run main.py. Does NOT train models. Does NOT download data.
#          All model output is pre-committed to output/ and served as-is.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps (needed by lightgbm / matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies FIRST (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (includes pre-committed output/ files)
COPY . .

# Streamlit config — suppress welcome screen & telemetry
RUN mkdir -p /root/.streamlit && printf '\
[general]\n\
email = ""\n\
[browser]\n\
gatherUsageStats = false\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
' > /root/.streamlit/config.toml

# Railway injects $PORT at runtime
EXPOSE 8501

CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
