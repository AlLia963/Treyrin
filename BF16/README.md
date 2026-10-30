---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-generation
---

# Treyrin v5 BF16

This directory provides two formats: a mixed BF16/FP32 Transformers checkpoint and native BF16 GGUF. The six Python files implement custom configuration, model loading, architecture bundling and state restoration. The GGUF carries its own structural source and cognitive programs.

`model.safetensors` contains 1,014 tensors, including the cognitive graph, 76 experience-state tensors and 5 Spark-state tensors. `Treyrin-v5-BF16.gguf` provides the native representation.

Download source-format weights from the repository root with `.\Download.ps1 -Variant Transformers`. Install dependencies from `BF16/requirements.txt` into a suitable Python environment, then load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./BF16', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    './BF16', trust_remote_code=True, local_files_only=True, device='cuda:0'
)
```

For native BF16 operation use `.\Run-Llama.ps1 -Variant BF16` from the repository root. The supplied extension implements Treyrin architecture support. BF16 requires more memory than Q8_0; CPU offload and context allocation affect speed. Q8_0 is the default local deployment.

See [architecture](../docs/ARCHITECTURE.md), [runtime setup](../runtime/README.md), and the [model card](../README.md) for execution boundaries, measurements and attribution.
