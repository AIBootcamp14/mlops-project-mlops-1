FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY ./api /app/api
COPY ./models /app/models
COPY . /app

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet stopwords

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
