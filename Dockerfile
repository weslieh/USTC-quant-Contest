# Container image for the quantcontest2026 training pipeline.
# Designed for managed training services (GCP Vertex AI custom jobs,
# Aliyun PAI-DLC, AWS SageMaker Training) and plain `docker run` alike.
#
# Data and outputs are exchanged via mounted volumes (the norm on managed
# platforms). Object-storage sync is optional and pluggable through env vars
# in scripts/train-entrypoint.sh, so the image stays lean and provider-agnostic.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_ROOT=/mnt/data \
    OUT_DIR=/mnt/output \
    TRAIN_ARGS=""

WORKDIR /workspace

# libgomp1: OpenMP runtime required by LightGBM's bundled .so (absent in -slim).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer; code changes won't bust it).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source in a single layer. .dockerignore keeps data/,
# model artifacts, .git/, and .claude/ out of the image.
COPY . .
RUN chmod +x scripts/train-entrypoint.sh

# Platforms mount input data at DATA_ROOT and read artifacts from OUT_DIR.
# Override the command/args on the platform if you need non-default flags.
ENTRYPOINT ["scripts/train-entrypoint.sh"]
CMD []
