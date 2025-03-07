FROM python:3.13.1-slim
# FROM dockerhub.timeweb.cloud/library/python:3.13.1-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir uv
RUN uv pip install --no-cache-dir -r requirements.txt --system
CMD [ "python", "./app.py" ]