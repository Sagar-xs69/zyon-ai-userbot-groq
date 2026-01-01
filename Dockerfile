FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for py-tgcalls, voice features, and deno for yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    build-essential \
    python3-dev \
    libffi-dev \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno for yt-dlp YouTube extraction
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="${DENO_INSTALL}/bin:${PATH}"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force upgrade yt-dlp to latest version (YouTube changes formats frequently)
RUN pip install --no-cache-dir --upgrade yt-dlp

# Copy YouTube cookies explicitly to ensure they are present
COPY youtube_cookies.txt .

# Copy application files
COPY . .

# Run the bot with unbuffered output
CMD ["python", "-u", "vc.py"]
