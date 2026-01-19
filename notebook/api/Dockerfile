FROM python:3.10.11-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY notebook/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire notebook folder
COPY notebook ./notebook

EXPOSE 10000

CMD ["gunicorn", "notebook.api.app:app", "--bind", "0.0.0.0:10000"]
