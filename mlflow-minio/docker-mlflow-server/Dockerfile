FROM python:3.9-slim-buster
COPY . /opt/app

WORKDIR /opt/app

RUN set -xe && \
    apt-get update 

RUN apt-get install -y apt-utils

RUN apt-get install -y libpq-dev

RUN apt-get install -y build-essential

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
