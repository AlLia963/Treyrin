---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-generation
---

# Treyrin v5 BF16

This directory contains two loading formats: the complete mixed BF16/FP32 Transformers checkpoint and the BF16 native GGUF. The six Python files implement the custom-code Transformers path. Native GGUF embeds its structural source and cognitive programs. See [format and execution details](../README.md#why-the-bf16-directory-contains-python).

`model.safetensors` contains 1,014 tensors, including the cognitive graph, 76 experience-state tensors and 5 Spark-state tensors. `Treyrin-v5-BF16.gguf` contains the native representation and embedded cognitive programs.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./BF16", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    "./BF16", trust_remote_code=True, local_files_only=True, device="cuda:0"
)
```

For native inference, run `Run-Llama.ps1 -Variant BF16` from the repository root. The supplied llama runtime includes Treyrin architecture support. BF16 weights, context cache and compute buffers require CPU/GPU memory; partial CPU offload reduces generation speed.

See the [model card](../README.md) for architecture, measurements and origin. This model uses the Qwen3.5-4B text decoder under Apache-2.0.
