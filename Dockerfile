FROM python:3.12.6-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U git+https://github.com/google-gemini/generative-ai-python@imagen
CMD [ "python", "./app.py" ]