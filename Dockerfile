FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
#RUN mkdir model

COPY app.py .

EXPOSE 12345

CMD ["python", "app.py"]