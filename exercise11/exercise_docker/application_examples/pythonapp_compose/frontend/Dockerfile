FROM python:3.8-alpine
RUN mkdir /app_frontend
ADD . /app_frontend
WORKDIR /app_frontend
RUN pip install -r requirements.txt
CMD ["python", "app.py"]