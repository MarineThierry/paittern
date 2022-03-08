FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

COPY api_wf /api_wf
# COPY /Users/Marine/code/Installation-GCP/wagon-bootcamp-337709-118bde2295fc.json /credentials.json
COPY paittern /paittern
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD uvicorn api_wf.fast:app --host 0.0.0.0 --port $PORT
