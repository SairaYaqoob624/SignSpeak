# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY signspeak/backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire signspeak folder
COPY signspeak/ /app/signspeak/

# Set working directory to signspeak/backend
WORKDIR /app/signspeak/backend

# Expose port 7860 (Hugging Face expects port 7860)
EXPOSE 7860

# Run Flask app using gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "live_gesture_server:app"]
