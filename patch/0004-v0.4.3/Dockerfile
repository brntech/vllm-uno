# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# Uno for vLLM v0.4.3 = the v0.4.1 release image + prompt lookup after Uno's drafts, with a per-request length gate, and
# startup checks that refuse UNO_DRAFT_MOE_TOPK=4 when draft graphs will not be captured (--enforce-eager, cudagraph mode
# NONE or PIECEWISE) or a reachable draft batch has no captured graph. Prompt lookup is OFF unless UNO_PLOOKUP_L is set,
# so an unconfigured v0.4.3 server runs the v0.4.2 path.
# Build from the repository root, so the image carries this release's own documentation:
#   docker build -f patch/0004-v0.4.3/Dockerfile -t vllm-uno:0.4.3 .
ARG BASE_IMAGE=ghcr.io/brntech/vllm-uno:0.4.1@sha256:f679879cada7b90d658246305801f2e9316e6f21f75cb7c3d7cd8fa37642ca34
FROM ${BASE_IMAGE}
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
ARG P=patch/0004-v0.4.3
COPY ${P}/uno_plookup.py ${VP}/v1/worker/gpu/spec_decode/uno_plookup.py
COPY ${P}/uno.py.diff ${P}/model_runner.py.diff ${P}/triton_attn.py.diff ${P}/triton_unified_attention.py.diff ${P}/serve.sh.diff ${P}/scheduler.py.diff ${P}/async_scheduler.py.diff ${P}/cudagraph_utils.py.diff /tmp/plookup/
RUN set -e; \
    patch --forward --batch ${VP}/v1/worker/gpu/spec_decode/uno.py /tmp/plookup/uno.py.diff; \
    patch --forward --batch ${VP}/v1/worker/gpu/model_runner.py /tmp/plookup/model_runner.py.diff; \
    patch --forward --batch ${VP}/v1/attention/backends/triton_attn.py /tmp/plookup/triton_attn.py.diff; \
    patch --forward --batch ${VP}/v1/attention/ops/triton_unified_attention.py /tmp/plookup/triton_unified_attention.py.diff; \
    patch --forward --batch /opt/uno-kit/release/serve.sh /tmp/plookup/serve.sh.diff; \
    patch --forward --batch ${VP}/v1/core/sched/scheduler.py /tmp/plookup/scheduler.py.diff; \
    patch --forward --batch ${VP}/v1/core/sched/async_scheduler.py /tmp/plookup/async_scheduler.py.diff; \
    patch --forward --batch ${VP}/v1/worker/gpu/cudagraph_utils.py /tmp/plookup/cudagraph_utils.py.diff; \
    find ${VP} -name '*.orig' -delete; \
    rm -rf /tmp/plookup; \
    bash -n /opt/uno-kit/release/serve.sh; \
    echo 0.4.3 > /opt/uno-kit/release/VERSION; \
    python3 -c "import vllm.v1.worker.gpu.spec_decode.uno as u, vllm.v1.worker.gpu.spec_decode.uno_plookup as p, vllm.v1.attention.backends.triton_attn as ta, vllm.v1.core.sched.async_scheduler as a, vllm.v1.worker.gpu.cudagraph_utils as cg; assert p.gated_num_spec(6, 2, 4096, 5000) == 4 and p.gated_num_spec(6, 2, 4096, 4096) == 6; print('vllm-uno 0.4.3 import ok', p.NMAX, p.MAX_L)"; \
    sha256sum ${VP}/v1/worker/gpu/spec_decode/uno.py ${VP}/v1/worker/gpu/spec_decode/uno_plookup.py ${VP}/v1/worker/gpu/model_runner.py \
      ${VP}/v1/attention/backends/triton_attn.py ${VP}/v1/attention/ops/triton_unified_attention.py /opt/uno-kit/release/serve.sh \
      ${VP}/v1/core/sched/scheduler.py ${VP}/v1/core/sched/async_scheduler.py ${VP}/v1/worker/gpu/cudagraph_utils.py > /opt/uno-plookup.sha256; \
    rm -rf /opt/uno-kit/docs /opt/uno-kit/README.md /opt/uno-kit/CHANGELOG.md /opt/uno-kit/CITATION.cff /opt/uno-kit/RELEASE-NOTES-*.md
# The kit text in /opt/uno-kit is this release's (the v0.4.1 base layers carry the v0.4.1 text): exactly the documents
# release/assets.json lists, plus this release's notes. Named, never a directory or a glob.
COPY README.md CHANGELOG.md CITATION.cff RELEASE-NOTES-0.4.0.md RELEASE-NOTES-0.4.1.md RELEASE-NOTES-0.4.2.md RELEASE-NOTES-0.4.3.md /opt/uno-kit/
COPY docs/PR-DESCRIPTION-GEMMA.md docs/configuration.md docs/validation.md /opt/uno-kit/docs/
LABEL org.opencontainers.image.version="0.4.3" \
      ai.uno.release.base="vllm-uno:0.4.1" \
      ai.uno.overlay="prompt-lookup + length gate (v0.4.3)"
