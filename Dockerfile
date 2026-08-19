FROM python:3.11-slim

# libgl1/libglib2.0-0 are needed for opencv even in headless mode (font/codec libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
