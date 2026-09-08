# Базовый образ Ubuntu 24.04 критически важен:
# скрипт требует glibc >= 2.38 (в 22.04 стоит 2.35) и штатный Python 3.12.
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей, соответствующих скрипту
RUN apt-get update && apt-get install -y \
    curl \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Создаем виртуальное окружение, чтобы точно повторить логику установки
# и избежать блокировки pip (externally-managed-environment) в Ubuntu 24.04
ENV VIRTUAL_ENV=/app/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -q --upgrade pip

# Загрузка и распаковка специфичной сборки
COPY vllm-2080ti-sm75-cu130-wheels.tar.gz ./
RUN tar xzf vllm-2080ti-sm75-cu130-wheels.tar.gz && \
    rm vllm-2080ti-sm75-cu130-wheels.tar.gz

# Установка Torch cu130 (тяжелый шаг)
RUN pip install -q torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# Установка зависимостей
RUN pip install -q flashinfer-python==0.6.16.post3 "transformers>=5.5.3" numpy

# Установка движка и хак для flash_qla (строго без разрешения зависимостей)
RUN pip install -q ./vllm-*.whl && \
    pip install -q --no-deps ./flash_qla-*.whl && \
    rm ./*.whl

EXPOSE 8000

# Путь к модели вынесен в ENV для удобной подмены через docker-compose
ENV MODEL_PATH="/models/Qwen3"

# Запуск с флагами, извлеченными из systemd-юнита (включая max-num-seqs 1)
CMD python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name Qwen3.8 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.94 \
    --max-num-seqs 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --tool-call-parser qwen3_xml

