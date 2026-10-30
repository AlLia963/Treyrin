---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
tags: [gguf, quantized]
---

# Treyrin v5 Q8_0

`Treyrin-v5-Q8_0.gguf` is the 4.22 GiB native model with Q8_0 language weights. Its architecture identifier is `treyrin`. It includes the tokenizer, cognitive parameters, ONNX programs, structural implementation and initial state. Cognitive tensors retain FP32 where required. The embedded lifecycle source still uses Python in the native runtime. See [format and execution details](../README.md#why-the-bf16-directory-contains-python).

Run `Run-Llama.ps1 -Variant Q8_0` from the repository root. The supplied llama runtime includes Treyrin architecture support. Q8_0 runs independently of the BF16 files. Quantization introduces numerical approximation and can change text and rankings.

See the [model card](../README.md) for architecture, measurements, runtime requirements and Qwen3.5-4B attribution.
