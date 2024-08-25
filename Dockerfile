FROM python:3.8

RUN mkdir /fast_pytorch_app

WORKDIR /fast_pytorch_app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

WORKDIR src

CMD gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000