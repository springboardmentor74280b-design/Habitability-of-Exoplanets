FROM python:3.10.11-slim

WORKDIR /app

COPY notebook/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY notebook/api/ .

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
