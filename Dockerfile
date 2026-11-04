FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt --system
CMD ["uv", "run", "./app.py" ]