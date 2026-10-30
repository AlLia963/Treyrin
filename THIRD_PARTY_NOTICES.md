# Third-party notices

The pretrained text decoder and tokenizer originate from Qwen/Qwen3.5-4B, Qwen team / Alibaba Cloud, commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, under Apache-2.0. The upstream vision encoder is excluded.

The native runtime includes llama.cpp/ggml (MIT), ONNX Runtime (MIT), Python (PSF), PyTorch CPU (BSD-3-Clause), NumPy (BSD-3-Clause), safetensors (Apache-2.0), and their runtime dependencies. Their distributed license and third-party notice files accompany the native runtime. NVIDIA CUDA redistributable components remain governed by NVIDIA's license. NVIDIA drivers are installed separately.

The optional Transformers path uses Transformers, PEFT, Accelerate, tokenizers and huggingface-hub (Apache-2.0), plus PyTorch, NumPy and psutil (BSD-3-Clause). Each dependency's distributed license applies.
