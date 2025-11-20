
<div align="center">
  <img src="assets/logo.png" width="120"/>
  <h1>TileRT: Tile-Based Runtime for<br>Ultra-Low-Latency LLM Inference</h1>
  <p>
    <a href="https://pypi.org/project/tilert/"><img src="https://img.shields.io/pypi/v/tilert" alt="PyPI version" height="20"></a>
    <a href="https://huggingface.co/Tile-AI"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face" height="20"></a>
  </p>
  <p>
    <a href="#installation"><b>Installation</b></a> |
    <a href="#getting-started"><b>Getting Started</b></a>
  </p>
</div>

TileRT is an experimental project that explores core compiler techniques designed to serve large language models in ultra-low-latency scenarios. Unlike existing inference systems built for high-throughput batch processing, TileRT focuses on delivering extreme responsiveness—critical for applications such as high-frequency trading, interactive AI, real-time decision-making, long-running agents, and AI coding, where users care more about the latency of a few requests or even a single request.

The goal of the TileRT project is to push the latency boundaries of LLMs without compromising model size or quality—for example, enabling models with hundreds of billions of parameters to run at millisecond-level TPOT.

<p align="center">
<img src="assets/generate.gif" alt="TileRT Benchmark"><br>
Fig. Sequence generation using SGLang (left), vLLM (middle), and TileRT (right) with the DeepSeek-V3.2-Exp model.
</p>

TileRT addresses these challenges with a new tile-level runtime engine. It uses a compiler-driven approach to decompose LLM operators into fine-grained tile-level tasks, and a tile-level runtime that reschedules compute, I/O, and communication across multiple devices in a highly overlapped manner. This allows TileRT to minimize idle time and maximize hardware utilization. These compiler techniques will be incorporated into TileLang and TileScale.

We evaluated TileRT’s preliminary performance using the DeepSeek-V3.2-Exp model (without lossy optimizations such as quantization or distillation) with a batch size of 1 on 8× NVIDIA B200 GPUs. As shown in the benchmark below, TileRT significantly outperforms existing inference systems:

<p align="center">
<img src="assets/perf.png" alt="TileRT Benchmark" width="400"><br>
Fig. Evaluation setup: Input seqlen/output seqlen: 1K/1K, SGLang-0.5.5, vLLM-0.11.0, CUDA-12.9
</p>

TileRT is a continuously evolving project. Our ongoing plans include pursuing more aggressive optimizations, supporting various batch sizes, more model families and more hardware, and establishing a new foundation for low-latency AI inference. Stay tuned for updates!
