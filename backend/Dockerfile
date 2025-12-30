FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn python-multipart

# Copy code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port (internal)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
