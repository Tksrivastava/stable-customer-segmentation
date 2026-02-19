# =====================================
# STAGE 1 — BUILD WHEEL
# =====================================
FROM python:3.11-slim AS builder

WORKDIR /build

# system deps needed for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# install build tool
RUN pip install --no-cache-dir build

# copy ONLY package source
COPY core/ ./core
COPY pyproject.toml .

# build wheel
RUN python -m build


# =====================================
# STAGE 2 — RUNTIME IMAGE
# =====================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /stable-customer-segmentation

# runtime system deps
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirement.txt

# copy wheel ONLY from builder
COPY --from=builder /build/dist/*.whl /tmp/

# install your package
RUN pip install --no-cache-dir /tmp/*.whl

# copy pipelines + scripts (NOT core source)
COPY . .

# ensure script executable
RUN chmod +x run.sh

ENTRYPOINT ["bash", "./run.sh"]