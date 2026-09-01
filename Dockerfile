# nvissues-postgres-astra — NVBugs retrieval UI on Astra (OpenShift).
#
#   docker build --platform linux/amd64 -t nvissues-postgres-astra .
#
# The platform builds this itself: astra-ci's bootstrap generates a
# .gitlab-ci.yml that looks for a Dockerfile at the repository root, which is
# why this lives here rather than under deploy/. Building locally is for
# catching errors before the pipeline does.
#
# Build for the cluster's architecture, not the laptop's. Astra runs OpenShift
# on x86 DGX B200s, so an arm64 image -- what a DGX Spark or an Apple laptop
# produces by default -- will pull and then fail to start.
#
# What the image contains and why
# -------------------------------
# No corpus data. All 999,198 bugs, 4.36M chunk vectors, facets and edges live
# in the internal Postgres, so adding or growing a corpus needs no rebuild and
# the image cannot go stale against the data.
#
# The query embedder, on the other hand, is baked in and has to be. Those 4.36M
# chunk vectors were produced by Qwen3-Embedding-0.6B; a query embedded by any
# other model lands in a different vector space and retrieval gets quietly
# worse rather than failing. Downloading 1.2 GB from huggingface.co at pod
# start would also put an external dependency on the critical path of every
# rollout, in a cluster that should not need one.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch, installed before anything else so transformers cannot pull
# the CUDA build in behind it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-Embedding-0.6B', \
allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])"

# From here on the container must never reach the hub. Loading the model once
# under that constraint turns an incomplete cache into a build failure rather
# than a pod that starts and then fails its first question.
ENV HF_HUB_OFFLINE=1
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True, padding_side='left'); \
AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)"

COPY *.py config.yaml ./
COPY static/ ./static/

# Question sets only: the examples the UI offers. The bugs they name are in
# Postgres, not here.
COPY data/ ./data/

# OpenShift runs the container as an arbitrary non-root UID in the root group.
# What makes that work is group ownership, not the USER line below -- the
# assigned UID will not be 1001. The process writes only to the HF cache, and
# only if something forces a re-download.
RUN chgrp -R 0 /app /opt/hf && chmod -R g=u /app /opt/hf
USER 1001

ENV GI_CONFIG=/app/config.yaml \
    QUESTIONS_FILE=/app/data/dw1m_questions.json \
    INDEX_DEVICE=cpu \
    GI_EMBED_DEVICE=cpu \
    PORT=8080

EXPOSE 8080

# The platform injects PORT, so it cannot be baked into the argv list.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
