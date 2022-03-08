FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

COPY api /api
# COPY /Users/Marine/code/Installation-GCP/wagon-bootcamp-337709-118bde2295fc.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
