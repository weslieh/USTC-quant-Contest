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

# Install dependencies first (cached layer; code changes won't bust it).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project source. Data/ is excluded via .dockerignore.
COPY src/ ./src/
COPY train.py eda.py eeda.py ./
COPY timeseries_api/ ./timeseries_api/
COPY examples/ ./examples/
COPY strategy/main.py ./strategy/main.py
COPY README.md .gitignore ./

COPY scripts/train-entrypoint.sh /usr/local/bin/train-entrypoint.sh
RUN chmod +x /usr/local/bin/train-entrypoint.sh

# Platforms mount input data at DATA_ROOT and read artifacts from OUT_DIR.
# Override the command/args on the platform if you need non-default flags.
ENTRYPOINT ["/usr/local/bin/train-entrypoint.sh"]
CMD []
