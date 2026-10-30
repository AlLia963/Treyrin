---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
tags: [gguf, quantized]
---

# Treyrin v5 Q8_0

`Treyrin-v5-Q8_0.gguf` is a 4,534,809,504-byte native model with architecture identifier `treyrin`. Language weights use Q8_0; cognitive tensors retain required precision. The GGUF carries tokenizer data, cognitive parameters, ONNX programs, structural source and initial state. Native lifecycle operations still use the supplied embedded Python environment.

From the repository root, run `.\Run-Llama.ps1`. Missing weights and the matching runtime are retrieved from Release assets, reassembled and verified. This variant operates independently of BF16 files. Unmodified official llama.cpp does not currently recognize this model architecture.

See [runtime setup](../runtime/README.md), [architecture](../docs/ARCHITECTURE.md), and the [model card](../README.md). Model weights/tokenizer derive from Qwen3.5-4B, with Treyrin additions and adaptations recorded in NOTICE. Quantization can change numerical results and generated text.
