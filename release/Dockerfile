# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:00972dfd72988942138a7a6089eaee08580210b8@sha256:d55cb6858435cda5ab080987213b4a6b6bfce14ca9e0ffa2ecfab2b222818497
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
USER root
WORKDIR /opt/uno-kit
LABEL org.opencontainers.image.title="Uno for vLLM" \
      org.opencontainers.image.source="https://github.com/brntech/vllm-uno" \
      org.opencontainers.image.version="0.3.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.uno.upstream.commit="00972dfd72988942138a7a6089eaee08580210b8" \
      ai.uno.head.commit="cf87916880b051e8782521dfe2afa12e0627e172"
# build.sh supplies only files named in release/assets.json.
COPY . .
# The CI image embeds the exact source repository at its build commit. Clone
# locally from it so an image build never contacts GitHub or guesses a source.
# The patch series is ordered: the Model Runner V2 base layer, then Gemma 4.
RUN git clone --shared --no-checkout /vllm-workspace /opt/uno-vllm-source \
    && git -C /opt/uno-vllm-source -c core.autocrlf=false checkout --detach 00972dfd72988942138a7a6089eaee08580210b8 \
    && python3 release/stack.py apply /opt/uno-vllm-source \
    && python3 release/stack.py overlay /opt/uno-vllm-source \
    && rm -rf /opt/uno-vllm-source
RUN python3 -c "import requests, numpy, huggingface_hub; import vllm; import vllm.v1.spec_decode.uno_noise as uno_noise; import vllm.v1.worker.gpu.spec_decode.uno as uno; import vllm.v1.worker.gpu.spec_decode.uno_draft_moe as uno_draft_moe; print(vllm.__version__, uno_noise.__file__, uno.__file__, uno_draft_moe.__file__)" \
    && python3 release/check.py
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT ["bash", "/opt/uno-kit/release/serve.sh"]
CMD []
