FROM continuumio/anaconda3
COPY . /app
WORKDIR /app
CMD python main.py
