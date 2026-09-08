# Third-party notices

## vLLM

This project distributes modifications for
[vLLM](https://github.com/vllm-project/vllm/tree/e962733e08d10f7ca65dac4df99e116460b8b174),
based on commit `e962733e08d10f7ca65dac4df99e116460b8b174`.
vLLM is licensed under the Apache License, Version 2.0. Copyright and
attribution notices in the modified upstream source are retained. See the
repository `LICENSE` file and the
[license at the pinned revision](https://github.com/vllm-project/vllm/blob/e962733e08d10f7ca65dac4df99e116460b8b174/LICENSE).

## SGLang

Portions of `vllm/v1/spec_decode/uno_sampler.py`,
`vllm/v1/spec_decode/uno_lora_overlap.py`, and
`vllm/v1/spec_decode/uno_nosync.py` are adapted from or follow the SGLang Uno
implementation at commit `2c05ed4e7776c876478f4b2db61acb12b9a27d01`,
including compact-support sampling, draft LoRA overlap, and cached LoRA graph
state.

SGLang is licensed under the Apache License, Version 2.0.

Copyright 2023-2024 SGLang Team

See the [pinned SGLang source](https://github.com/sgl-project/sglang/tree/2c05ed4e7776c876478f4b2db61acb12b9a27d01)
and its [license](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/LICENSE).

## Uno and IFM

The Uno method and adapters are the work of IFM and the Uno paper authors.
This repository does not redistribute model weights. See the
[IFM Uno implementation](https://github.com/ifm-ai/uno), its
[third-party notice](https://github.com/ifm-ai/uno/blob/main/NOTICE), and the
paper, [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010).

## GSM8K

The validation prompt data includes one question from the
[GSM8K dataset](https://github.com/openai/grade-school-math), which is licensed
under the MIT License.

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
