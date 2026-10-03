<!-- markdownlint-disable MD001 MD041 -->
## About this fork

vLLM on an Intel Arc Pro B70 (XPU) serving Gemma 4 31B (EXL3) to many concurrent long-context chat sessions, based on upstream [`568afb3`](https://github.com/vllm-project/vllm/commit/568afb3a13806beb53bb2e6bd518269357b237c0).

| Decode | Baseline | This fork |
| --- | --- | --- |
| 1 session | 28.9 tok/s | 46.1 tok/s |
| 1 session, 19.5k context | 9.5 tok/s | 41.1 tok/s |
| 8 sessions, total | 53 tok/s | 166–181 tok/s |

Decode: XPU decode graphs (`FULL_DECODE_ONLY`), 2 MTP draft tokens with a pruned draft vocabulary, a 6-bit EXL3 `lm_head`, `--max-num-seqs 8`; baseline = eager decode, 3 draft tokens, bf16 head, 4 sequences.

| Concurrent sessions | Median time to first token (baseline → fork) | Median decode per session | Total generation |
| --- | --- | --- | --- |
| 4 | 20.3 s → 1.2 s | 14 → 23.5 tok/s | 25 → 33 tok/s |
| 6 | 19.4 s → 15.3 s | 8.7 → 21.7 tok/s | 24 → 41.5 tok/s |
| 8 | 66.8 s → 7.2 s | 7.9 → 21.4 tok/s | 28 → 57 tok/s |
| 10 | 104 s → 9.8 s | 8.7 → 20.1 tok/s | 24 → 57 tok/s |
| 12 | 136 s → 18 s | 9.0 → 20.1 tok/s | 27.5 → 67 tok/s |

Setup: about 19.5k tokens of context per session, `--max-num-seqs 4`, 16 GiB RAM cache; baseline = default `OffloadingConnector` + default sliding-window caching.

**Changes**

- `c22f9d8`: `VLLM_PREFIX_CACHE_REPLAY_SLACK_TOKENS` keeps sliding-window tails up to that many tokens before the replay boundary, so follow-up chat turns still hit the prefix cache under `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`.
- `46e4f7b`: an XPU copy backend (`swap_blocks_batch`) for `SimpleCPUOffloadConnector`, including its lazy mode, which previously needed CUDA/HIP batch copies.
- `af5173e`, `0dbd2e1`: on XPU the CPU cache is pinned in power-of-two chunks (24 GiB = 16 + 8), since PyTorch rounds each pinned allocation up to a power of two.
- `04515cc`: Gemma 4 reads its layer scalars at load instead of with `.item()` inside the compiled forward, so XPU decode graphs capture (`1eab1a7` first imports the image's Intel `gemma4.py`).
- `49eb1db`: `VLLM_GEMMA4_MTP_DRAFT_VOCAB` (a JSON list of token ids) limits the MTP drafter to those tokens; the target still verifies with its full head.

**Enabling it**

```bash
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 \
VLLM_PREFIX_CACHE_REPLAY_SLACK_TOKENS=256 \
vllm serve ... \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":<bytes>,"lazy_offload":true}}'
```

Keep `cpu_bytes_to_use` well under free RAM: it is pinned at startup and cannot be swapped. On Intel, also set `NEOReadDebugKeys=1 TreatNonUsmForTransfersAsSharedSystem=0 EnableSharedSystemUsmSupport=0`: without them the copy engine can fault reading swapped host pages while loading weights.

Tested on a real B70: vLLM's prefix-caching and KV-cache-manager tests plus a new XPU copy-backend test pass, and temperature-0 outputs match within normal prefix-cache noise. Not yet proposed upstream.

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
