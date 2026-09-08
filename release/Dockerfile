# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=vllm/vllm-openai@sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
USER root
WORKDIR /opt/uno-kit
LABEL org.opencontainers.image.title="Uno for vLLM" \
      org.opencontainers.image.source="https://github.com/brntech/vllm-uno" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.uno.upstream.commit="e962733e08d10f7ca65dac4df99e116460b8b174"
# build.sh supplies only files named in release/assets.json.
COPY . .
# Preserve the upstream compiled libraries and matching dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 release/stack.py apply /opt/uno-vllm-source \
    && python3 release/stack.py overlay /opt/uno-vllm-source \
    && rm -rf /opt/uno-vllm-source
RUN python3 -c "import requests, numpy, huggingface_hub; import vllm; from vllm.v1.spec_decode import uno; print(vllm.__version__, uno.__file__)" \
    && python3 release/check.py
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT ["bash", "/opt/uno-kit/release/serve.sh"]
CMD []
