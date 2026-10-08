# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# Uno for vLLM v0.4.4 on AMD Radeon (ROCm, gfx1201: Radeon AI PRO R9700), built from the stock vLLM v0.30.0 ROCm image.
#   vllm-v0.30.0-rocm.diff   Uno v0.4.4 (patches 0001-0005 of this repository) ported onto vLLM v0.30.0, plus three
#                            ROCm admissions (uno_rocm.py: Uno, its 4-expert drafts and the AWQ MoE contract admit
#                            gfx1201 after the CUDA condition fails), the reader of the gfx12 W4A16 dispatch table and
#                            the LoRA kernel-config loader (the GPU name ROCm reports, exact-shape matches)
#   *.json                   the data those read: the gfx12 W4A16 dispatch table, the R9700 LoRA shrink/expand configs,
#                            and the R9700 fused-MoE config of Gemma 4 26B A4B's AWQ experts
#   serve.sh.diff            the launcher: v0.4.4's (release/serve.sh + the 0004 and 0005 diffs) with, on gfx12 GPUs,
#                            the adapter on the main stream, the MoE config folder by default and the prefill tiles
#   gfx12_prefill_tiles.py   larger attention tiles for prefill-shaped launches, patched in at container start on gfx12
# Python and data only; no compiled library changes (asserted). The build refuses another base: every file the diff
# changes must have v0.30.0's bytes, and every engine file it leaves must equal engine-files.sha256.
# Build from the repository root:
#   docker build -f patch/rocm-0.4.4/Dockerfile -t vllm-uno:0.4.4-rocm .
ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.30.0@sha256:2e7da1ad1c66836802072588adea75f9f4991da5f9545b4318e91d422c22ce6a
FROM ${BASE_IMAGE}
USER root
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG P=patch/rocm-0.4.4
COPY ${P}/vllm-v0.30.0-rocm.diff ${P}/base.sha256 ${P}/engine-files.sha256 ${P}/serve.sh.diff \
     ${P}/rdna_w4a16_gfx12_table.json ${P}/AMD_Radeon_R9700_SHRINK.json ${P}/AMD_Radeon_R9700_EXPAND_TRUE.json /tmp/rocm/
COPY patch/0004-v0.4.3/serve.sh.diff /tmp/rocm/0004-serve.sh.diff
COPY patch/0005-v0.4.4/serve.sh.diff /tmp/rocm/0005-serve.sh.diff
COPY release/serve.sh release/gemma4-draft-vocab-65536.json ${P}/gfx12_prefill_tiles.py /opt/uno-kit/release/
COPY ${P}/E=128,N=704,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json /opt/uno-kit/release/gfx12-moe/
RUN set -e; \
    cd ${SP}; sha256sum -c --quiet /tmp/rocm/base.sha256; \
    cut -c67- /tmp/rocm/base.sha256 > /tmp/rocm/base.list; \
    for f in $(cut -c67- /tmp/rocm/engine-files.sha256); do \
      if ! grep -qxF "$f" /tmp/rocm/base.list && [ -e "$f" ]; then echo "refuse: $f exists in the base image"; exit 3; fi; \
    done; \
    find ${SP} -name '*.so' -print0 | sort -z | xargs -0 sha256sum > /tmp/rocm/so-before.sha256; \
    patch -p1 --forward --batch -d ${SP} < /tmp/rocm/vllm-v0.30.0-rocm.diff; \
    cp /tmp/rocm/rdna_w4a16_gfx12_table.json ${SP}/vllm/model_executor/kernels/linear/mixed_precision/; \
    mkdir -p ${SP}/vllm/lora/ops/triton_ops/configs; \
    cp /tmp/rocm/AMD_Radeon_R9700_SHRINK.json /tmp/rocm/AMD_Radeon_R9700_EXPAND_TRUE.json ${SP}/vllm/lora/ops/triton_ops/configs/; \
    [ -z "$(find ${SP}/vllm \( -name '*.orig' -o -name '*.rej' \) -print -quit)" ]; \
    for f in $(cut -c67- /tmp/rocm/engine-files.sha256); do rm -f "$(dirname "$f")/__pycache__/$(basename "$f" .py)".*.pyc; done; \
    sha256sum -c --quiet /tmp/rocm/engine-files.sha256; \
    find ${SP} -name '*.so' -print0 | sort -z | xargs -0 sha256sum | cmp - /tmp/rocm/so-before.sha256; \
    cd /opt/uno-kit/release; \
    patch --forward --batch serve.sh /tmp/rocm/0004-serve.sh.diff; \
    patch --forward --batch serve.sh /tmp/rocm/0005-serve.sh.diff; \
    patch --forward --batch serve.sh /tmp/rocm/serve.sh.diff; \
    rm -f serve.sh.orig; bash -n serve.sh; \
    echo 0.4.4-rocm > VERSION; \
    python3 -c "import vllm, vllm.v1.spec_decode.uno_noise, vllm.v1.worker.gpu.spec_decode.uno, vllm.v1.worker.gpu.spec_decode.uno_draft_moe, vllm.v1.worker.gpu.spec_decode.uno_rocm as r, vllm.v1.worker.gpu.spec_decode.uno_plookup as p, vllm.v1.attention.backends.triton_attn as b, vllm.v1.attention.ops.triton_unified_attention as o, vllm.v1.core.sched.async_scheduler, vllm.v1.worker.gpu.cudagraph_utils, vllm.v1.worker.gpu.model_runner, vllm.v1.worker.gpu_worker; from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as rh; import vllm.lora.ops.triton_ops.utils as lu; assert p.gated_num_spec(6, 2, 4096, 5000) == 4 and p.gated_num_spec(6, 2, 4096, 4096) == 6; assert not b._uno_splitkv_multi_enabled() and not o._uno_splitkv_multi_enabled(); assert r.VALIDATED_ROCM_ARCHS == {'gfx1201'}, r.VALIDATED_ROCM_ARCHS; assert rh._GFX12_TABLE, 'W4A16 table not loaded'; f = sorted(q.name for q in lu._BUILTIN_CONFIG_DIR.iterdir()); assert f == ['AMD_Radeon_R9700_EXPAND_TRUE.json', 'AMD_Radeon_R9700_SHRINK.json'], f; print('vllm-uno 0.4.4-rocm import ok', vllm.__version__, rh._GFX12_TABLE_SHA, f)"; \
    UNO_GPU_ARCH=gfx12 UNO_PROFILE=gemma4 UNO_RECOMMENDED=1 UNO_DRY_RUN=1 bash serve.sh m /tmp 2>&1 >/dev/null | grep -qx 'Uno settings: UNO_RECOMMENDED=1 UNO_K=5 UNO_PLOOKUP_L=2 UNO_PLOOKUP_MAX_CTX=4096 UNO_DRAFT_MOE_TOPK=4 UNO_GEMMA_SPLITKV=1 UNO_GEMMA_SPLITKV_MULTI=1 capture_sizes=\[1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,28,32,40,48,56,64\]' || { echo 'UNO_RECOMMENDED dry run failed'; exit 1; }; \
    UNO_GPU_ARCH=gfx12 UNO_PROFILE=gemma4 UNO_RECOMMENDED=1 UNO_DRY_RUN=1 bash serve.sh m /tmp 2>/dev/null | grep -q '^env VLLM_TUNED_CONFIG_FOLDER=/opt/uno-kit/release/gfx12-moe VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LORA_ENABLE_DUAL_STREAM=0 ' || { echo 'gfx12 defaults (MoE config folder, single-stream LoRA) missing from the dry run'; exit 1; }; \
    sed "s#  #  ${SP}/#" /tmp/rocm/engine-files.sha256 > /opt/uno-runtime.sha256; \
    sha256sum /opt/uno-kit/release/serve.sh /opt/uno-kit/release/gfx12_prefill_tiles.py /opt/uno-kit/release/gemma4-draft-vocab-65536.json \
      /opt/uno-kit/release/gfx12-moe/*.json >> /opt/uno-runtime.sha256; \
    sha256sum -c --quiet /opt/uno-runtime.sha256; \
    rm -rf /tmp/rocm
ENV HF_HOME=/root/.cache/huggingface PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.title="Uno for vLLM (ROCm)" \
      org.opencontainers.image.version="0.4.4-rocm" \
      ai.uno.base="vllm/vllm-openai-rocm:v0.30.0" \
      ai.uno.overlay="Uno v0.4.4 on vLLM v0.30.0 ROCm: gfx1201 admissions, W4A16 table, LoRA and MoE configs, prefill tiles"
EXPOSE 8000
ENTRYPOINT ["bash", "/opt/uno-kit/release/serve.sh"]
CMD []
