FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git
RUN python3 -m pip install --no-cache-dir fastapi uvicorn python-multipart

COPY app.py .

EXPOSE 8000
CMD ["python3", "app.py"]
