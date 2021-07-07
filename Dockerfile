# syntax=docker/dockerfile:1
FROM python:3.7-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
COPY . /code/


