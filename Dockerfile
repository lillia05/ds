# Base image Python
FROM python:3.12-slim

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements dulu
COPY requirements.txt .

# Install dependencies tanpa cache biar ringan
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project kamu ke container
COPY . .

# Buka port 8000 untuk Railway
EXPOSE 8000

# Jalankan Flask menggunakan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]