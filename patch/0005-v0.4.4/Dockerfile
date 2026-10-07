# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# Uno for vLLM v0.4.4 = the v0.4.3 release image + split-KV attention for decode steps with several requests in flight
# (UNO_GEMMA_SPLITKV_MULTI=1), capture sizes 48/56/64 that come with it, and UNO_RECOMMENDED=1 in the launcher, which
# turns on the recommended settings in one switch. With neither variable set, a v0.4.4 server runs the v0.4.3 path.
# The build fails unless UNO_RECOMMENDED=1 prints the settings line and the engine command (sha256) of the measured launch.
# Build from the repository root, so the image carries this release's own documentation:
#   docker build -f patch/0005-v0.4.4/Dockerfile -t vllm-uno:0.4.4 .
ARG BASE_IMAGE=ghcr.io/brntech/vllm-uno:0.4.3@sha256:908e3ba1ec3c3e9d6f2c442c5a6995c376fa8ad52758724de4aaa5e42736bd17
FROM ${BASE_IMAGE}
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
ARG P=patch/0005-v0.4.4
COPY ${P}/triton_attn.py.diff ${P}/triton_unified_attention.py.diff ${P}/serve.sh.diff /tmp/v044/
RUN set -e; \
    patch --forward --batch ${VP}/v1/attention/backends/triton_attn.py /tmp/v044/triton_attn.py.diff; \
    patch --forward --batch ${VP}/v1/attention/ops/triton_unified_attention.py /tmp/v044/triton_unified_attention.py.diff; \
    patch --forward --batch /opt/uno-kit/release/serve.sh /tmp/v044/serve.sh.diff; \
    find ${VP} /opt/uno-kit -name '*.orig' -delete; \
    rm -rf /tmp/v044; \
    bash -n /opt/uno-kit/release/serve.sh; \
    echo 0.4.4 > /opt/uno-kit/release/VERSION; \
    python3 -c "import vllm.v1.attention.backends.triton_attn as b, vllm.v1.attention.ops.triton_unified_attention as o, vllm.v1.worker.gpu.spec_decode.uno as u; assert not b._uno_splitkv_multi_enabled() and not o._uno_splitkv_multi_enabled(); print('vllm-uno 0.4.4 import ok, multi-request split-KV off unless UNO_GEMMA_SPLITKV_MULTI is set')"; \
    UNO_PROFILE=gemma4 UNO_RECOMMENDED=1 UNO_DRY_RUN=1 bash /opt/uno-kit/release/serve.sh m /tmp 2>&1 >/dev/null | grep -qx 'Uno settings: UNO_RECOMMENDED=1 UNO_K=5 UNO_PLOOKUP_L=2 UNO_PLOOKUP_MAX_CTX=4096 UNO_DRAFT_MOE_TOPK=4 UNO_GEMMA_SPLITKV=1 UNO_GEMMA_SPLITKV_MULTI=1 capture_sizes=\[1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,28,32,40,48,56,64\]' ||{ echo 'UNO_RECOMMENDED dry run failed'; exit 1; }; \
    [ "$(env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root LANG=C.UTF-8 UNO_PROFILE=gemma4 UNO_RECOMMENDED=1 UNO_DRY_RUN=1 bash /opt/uno-kit/release/serve.sh m /tmp 2>/dev/null | sha256sum | cut -d' ' -f1)" = 3eaf58c655fc6b41c726369360ea1e832db7a148ec0e9eafd78bf63d0c8ea75d ] || { echo 'UNO_RECOMMENDED=1 no longer launches the measured engine command'; exit 1; }; \
    rm -f /opt/uno-plookup.sha256; \
    sha256sum ${VP}/v1/worker/gpu/spec_decode/uno.py ${VP}/v1/worker/gpu/spec_decode/uno_plookup.py ${VP}/v1/worker/gpu/model_runner.py \
      ${VP}/v1/attention/backends/triton_attn.py ${VP}/v1/attention/ops/triton_unified_attention.py /opt/uno-kit/release/serve.sh \
      ${VP}/v1/core/sched/scheduler.py ${VP}/v1/core/sched/async_scheduler.py ${VP}/v1/worker/gpu/cudagraph_utils.py > /opt/uno-runtime.sha256; \
    rm -rf /opt/uno-kit/docs /opt/uno-kit/README.md /opt/uno-kit/CHANGELOG.md /opt/uno-kit/CITATION.cff /opt/uno-kit/RELEASE-NOTES-*.md
# The kit text in /opt/uno-kit is this release's (the base layers carry the v0.4.3 text): exactly the documents
# release/assets.json lists, plus the notes of 0.4.2 to 0.4.4. Named, never a directory or a glob.
COPY README.md CHANGELOG.md CITATION.cff RELEASE-NOTES-0.4.0.md RELEASE-NOTES-0.4.1.md RELEASE-NOTES-0.4.2.md RELEASE-NOTES-0.4.3.md RELEASE-NOTES-0.4.4.md /opt/uno-kit/
COPY docs/PR-DESCRIPTION-GEMMA.md docs/configuration.md docs/validation.md /opt/uno-kit/docs/
LABEL org.opencontainers.image.version="0.4.4" \
      ai.uno.release.base="vllm-uno:0.4.3" \
      ai.uno.overlay="multi-request split-KV + capture sizes to 64 + UNO_RECOMMENDED (v0.4.4)"
