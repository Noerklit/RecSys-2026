FROM python:3.8-alpine
RUN mkdir /app_backend
ADD . /app_backend
WORKDIR /app_backend
RUN pip install -r requirements.txt
CMD ["python", "app.py"]