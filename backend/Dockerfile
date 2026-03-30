# Use a lightweight Python 3.11 image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Create a temporary uploads folder and give it read/write permissions
ENV TEMP_UPLOADS_DIR=/app/temp_uploads
RUN mkdir -p temp_uploads && chmod 777 temp_uploads

# Expose the default Hugging Face port
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]