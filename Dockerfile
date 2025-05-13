FROM python:3.13.3-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir uv
RUN uv pip install --no-cache-dir -r requirements.txt --system
CMD ["uv", "run", "./app.py" ]