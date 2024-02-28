FROM python:3.12.2-alpine

WORKDIR /app

COPY ./api_keys.json /app
RUN git clone https://github.com/ficus2904/bards-know.git .
RUN pip install --no-cache-dir -r requirements.txt


CMD [ "python", "./app.py" ]