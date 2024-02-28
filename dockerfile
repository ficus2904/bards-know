FROM python:3.12.2-alpine

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "./app.py" ]