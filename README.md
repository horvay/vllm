<!-- markdownlint-disable MD001 MD041 -->
## About this fork

This is a fork of vLLM for Neverending Quest (NQ), a local AI tabletop Game Master app. NQ serves Gemma 4 31B (EXL3) on an Intel Arc Pro B70 to several players at once. The branch `nq/xpu-kv-offload` is based on upstream commit [`568afb3`](https://github.com/vllm-project/vllm/commit/568afb3a13806beb53bb2e6bd518269357b237c0), the commit the exl3xpu image (`ghcr.io/0xsero/exl3xpu`, an Intel Arc / XPU build of vLLM with EXL3 kernels) was built from. It adds two commits on top.

**1. Keep sliding-window tails just before the replay boundary** (`c22f9d8`, `vllm/v1/core/single_type_kv_cache_manager.py`). Adds the environment variable `VLLM_PREFIX_CACHE_REPLAY_SLACK_TOKENS` (default `0`, which changes nothing). It only acts when upstream's `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` is set. With a retention interval of `0`, a sliding-window group caches only the tail at a prompt's replay boundary (`num_prompt_tokens - 1`). A follow-up chat turn diverges a little earlier, where the assistant reply starts, and its prefix hit rounds down to the scheduler block. When that lands one block earlier, no tail matches and the whole prompt is recomputed. The slack also keeps the tails ending up to that many tokens before the boundary, one per scheduler block. NQ uses `256`.

**2. SimpleCPUOffloadConnector copy backend for Intel XPU** (`46e4f7b`, `vllm/v1/simple_kv_offload/copy_backend.py` and `worker.py`). `SimpleCPUOffloadConnector`, including its lazy mode, copied blocks only through `cuMemcpyBatchAsync` / `hipMemcpyBatchAsync`. On XPU it now uses a new `XpuCopyBackend`, which runs vLLM's `swap_blocks_batch` op (the copy path `OffloadingConnector` already uses on XPU) on a background thread. XPU has no `cudaHostRegister`, so the CPU cache is one pinned buffer sliced per KV tensor, capped to the largest power of two that fits the requested size (PyTorch's caching host allocator rounds pinned allocations up to a power of two). Load and store streams come from `current_platform`.

**Why both matter for Gemma 4.** 50 of Gemma 4's 60 layers use a 1024-token sliding window, about 90% of the per-token KV cache. At vLLM's defaults every past sliding-window block stays cached and goes to the recent end of the eviction queue, so a new long prompt pushes other conversations' useful blocks off the GPU. The default `OffloadingConnector` also copies every chunk to RAM as it is computed: about 340 KB per token measured, against about 1.2 GB actually needed per 20k-token context. With `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` and the replay slack, only reusable tails are cached. With the lazy `SimpleCPUOffloadConnector`, blocks move to RAM only when they are about to leave the GPU, and uncached blocks are never moved.

**How NQ enables it:**

```bash
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 \
VLLM_PREFIX_CACHE_REPLAY_SLACK_TOKENS=256 \
vllm serve ... \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":<bytes>,"lazy_offload":true}}'
```

NQ's installer overlays only the three changed files onto the exl3xpu image, pinned to commit `46e4f7b7adbc1fadb2ca112574d7094b671ac807` and checked by SHA-256, after confirming that the image's copies match upstream at the base commit.

**Testing.** The new tests (a replay-slack test in `tests/v1/core/test_prefix_caching.py` and `tests/v1/simple_kv_offload/test_xpu_copy_backend.py`), together with the rest of `tests/v1/core/test_prefix_caching.py` and `tests/v1/core/test_single_type_kv_cache_manager.py`, pass inside the exl3xpu image on a real Arc B70. The `tests/v1/simple_kv_offload/test_scheduler.py` tests that download `facebook/opt-125m` from Hugging Face fail offline, identically on unmodified vLLM. In a real-engine check at temperature 0, fresh runs are bit-identical. A prompt restored from RAM matched the GPU-cached result within the noise floor that GPU prefix caching itself shows on this engine (up to about 0.14 logprob drift, because the fp8 KV cache is read back while a fresh prefill attends at full precision).

**Measured effect.** Early load-test numbers: 4 simulated players at about 19.5k tokens of context each, Arc B70, 4 sequences, 16 GiB RAM cache, compared with the default offloader. The full ramp results are still pending.

| Metric | Default offloader | This fork |
| --- | --- | --- |
| Median time to first token | 20.3 s | 1.4 s |
| Per-player decode speed | 14 tok/s | 24 tok/s |
| GPU prefix cache hit rate | 7% | 54% |
| RAM cache hit rate | 15% | 43% |

**Upstream status.** Not proposed upstream yet. Before an upstream PR, `VLLM_PREFIX_CACHE_REPLAY_SLACK_TOKENS` should move into `vllm/envs.py` (it reads `os.environ` directly here because the exl3xpu image ships a modified `envs.py`), and the commits need DCO sign-off by the author.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
