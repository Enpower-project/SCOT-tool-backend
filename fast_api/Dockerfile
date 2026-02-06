FROM python:3.12.3-slim

RUN pip3 install --upgrade pip

COPY ./apisrc /app/src
COPY ./requirements_api.txt /app

#COPY ./cert3.pem /app
#COPY ./privkey3.pem /app
WORKDIR /app

RUN pip3 install -r requirements_api.txt

#EXPOSE 8000
ENV PYTHONPATH=/app/src

#NO SSL
#CMD ["uvicorn", "src.service:app", "--host=0.0.0.0", "--reload"]


#SSL
#CMD ["uvicorn", "src.service:app","--ssl-keyfile", "privkey3.pem", "--ssl-certfile","cert3.pem", "--host=0.0.0.0", "--reload"]

# uvicorn service:app --host=0.0.0.0 --reload
