# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    XDG_DATA_HOME=/opt/bincomplete/share

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -c requirements.txt setuptools wheel \
    && python -m pip install --no-cache-dir -c requirements.txt torch --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=bind,target=/src,rw \
    python -m pip install --no-cache-dir --no-build-isolation -c requirements.txt "/src[sets,embeddings]" \
    && python -m pip check

COPY LICENSE THIRD_PARTY.md /usr/share/doc/bincomplete/
RUN HF_HOME=/tmp/bincomplete-model-download bincomplete models install dnabert_s \
    && rm -rf /tmp/bincomplete-model-download \
    && chmod -R a+rX /opt/bincomplete/share \
    && cp /usr/share/common-licenses/Apache-2.0 /usr/share/doc/bincomplete/DNABERT-S-LICENSE \
    && useradd --create-home --uid 10001 --user-group bincomplete \
    && install -d -o bincomplete -g bincomplete /work

ENV BINCOMPLETE_CACHE=/tmp/bincomplete-cache \
    HF_HOME=/tmp/bincomplete-huggingface \
    XDG_CONFIG_HOME=/tmp/bincomplete-config

WORKDIR /work
USER bincomplete
ENTRYPOINT ["bincomplete"]
CMD ["--help"]
