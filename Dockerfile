# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FORCE_IMPORT_DEPENDENCIES=1

# Install system dependencies
# ffmpeg is required for audio processing
# git is often needed for installing pip packages from source
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Increase timeout for large downloads (like torch)
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Create directories for uploads and results if they don't exist
RUN mkdir -p uploads transcripciones

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run web_server.py when the container launches
CMD ["uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000"]
