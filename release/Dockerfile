# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:b87339888d29329c42c42573e34cc2beebdcc48b@sha256:e3ab6a1f24248420800ceeede9f14e605e72f4e2ebd48a91821db8c07708e0f5
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
USER root
WORKDIR /opt/uno-kit
LABEL org.opencontainers.image.title="Uno for vLLM" \
      org.opencontainers.image.source="https://github.com/brntech/vllm-uno" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.uno.upstream.commit="b87339888d29329c42c42573e34cc2beebdcc48b"
# build.sh supplies only files named in release/assets.json.
COPY . .
# The CI image embeds the exact source repository at its build commit. Clone
# locally from it so an image build never contacts GitHub or guesses a source.
RUN git clone --shared --no-checkout /vllm-workspace /opt/uno-vllm-source \
    && git -C /opt/uno-vllm-source -c core.autocrlf=false checkout --detach b87339888d29329c42c42573e34cc2beebdcc48b \
    && python3 release/stack.py apply /opt/uno-vllm-source \
    && python3 release/stack.py overlay /opt/uno-vllm-source \
    && rm -rf /opt/uno-vllm-source
RUN python3 -c "import requests, numpy, huggingface_hub; import vllm; import vllm.v1.spec_decode.uno_noise as uno_noise; import vllm.v1.worker.gpu.spec_decode.uno as uno; print(vllm.__version__, uno_noise.__file__, uno.__file__)" \
    && python3 release/check.py
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT ["bash", "/opt/uno-kit/release/serve.sh"]
CMD []
