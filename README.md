---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
tags: [gguf, custom-code, stateful, decision-model]
---

# Treyrin v5

Treyrin is a stateful language-and-decision model built from the Qwen3.5-4B text decoder and a trainable recurrent cognitive core. Four decoder-layer connections let language representations, experience, candidate judgments and goals participate in shared computation. The project targets document reading, research records and everyday assistance on local hardware, with continuing experience and revisable hypotheses.

This release contains the selected `v4-mix-f` parameter lineage, packaged as v5. It provides Q8_0 and BF16 GGUF models, a Transformers checkpoint, architecture source and a Windows runtime. The current GGUF architecture is `treyrin`: it requires the supplied llama.cpp extension. **Unmodified upstream llama.cpp b12000 fails to load this artifact with `unknown model architecture: 'treyrin'`.**

## Run the model

Requirements for the supplied binary package: Windows x64, PowerShell 5.1 or later, Git, `curl.exe`, and an NVIDIA driver compatible with its CUDA 13.3 runtime. Reserve approximately 20 GB of free disk space for Q8_0, downloaded transport parts, runtime and working files. Memory requirements depend on context length and CPU/GPU offload; the default launcher permits automatic fitting.

```powershell
git clone https://github.com/AlLia963/Treyrin.git
Set-Location Treyrin
.\Run-Llama.ps1
```

The launcher downloads Q8_0 and the versioned runtime from the `v5` Release, checks SHA256, installs dependencies, and starts `http://127.0.0.1:18888`. Model weights exceed ordinary Git file limits and are distributed as Release transport parts; the scripts reconstruct the original files. A Git clone contains their metadata and source, not the large weights.

Both an existing local checkout and a new clone use this same launcher and runtime artifact. Runtime, logs and individual state are writable directories outside the repository. Setup, parameter overrides, HTTP examples and stopping instructions are in [runtime/README.md](runtime/README.md).

## Problems addressed

The project investigates several related problems:

- Combining direct candidate judgment and open language through shared model computation, rather than relying on separately selected answers.
- Forming useful, revisable intuitions and hypotheses when evidence is incomplete.
- Continuing a single individual's work across long histories, with source identity, corrections and persistent learned state.
- Keeping internal hypotheses, intentions and goals distinct from observed facts.
- Learning from experience while retaining pretrained abilities and earlier experience.
- Making goal/action choice and initiative part of model computation, while a host provides execution opportunities and tools.
- Preserving cognitive methods and experience contracts when a future language backbone replaces the current small model.

The original use cases, design requirements and delivered/remaining scope are recorded in [OBJECTIVES.md](docs/OBJECTIVES.md). The initial plan also considered future cooperation with the AlLia and Marigold projects; those integrations are outside this model release.

## What Treyrin adds

| Component | Computation and purpose | Implementation |
|---|---|---|
| Layer-interleaved cognitive feedback | Projects decoder states into a shared core, updates recurrent state, and returns the result to later language computation | Visits after zero-based layers 7, 15, 23 and 31; width-32 core |
| Regional recurrent circuit | Separates content relay, recurrent cortical processing, competing selection and gating into connected trainable states | Cortical IT/PT streams; thalamic and basal-ganglia-style populations |
| Predictive and modulatory pathways | Uses prediction error, valence, modulatory state and drives to influence core updates | Cerebellar-style prediction, DA/ACh/NE modulation and drive readouts |
| Episodic admission and retrieval | Chooses whether/where to write an experience and retrieves it in a query-, source- and goal-dependent manner | Learned skip/replace policy, competitive retrieval, consolidation and code attention |
| Typed experience and mental workspace | Records provenance and revision relationships, with separate treatment of observations, hypotheses, intentions and goals | Neural state, mental archive, source journal and paged storage |
| Native judgment and Spark | Forms recurrent alternatives, ranks candidate descriptions with the same model, and stores/revises explicit hypotheses | Adaptive candidate population plus same-body lifecycle calls |
| Joint goal/action control | Selects a goal/action pair using goal semantics, elapsed time, resource and interruption context | One normalized joint head with permission/resource masks |
| Individual learning and recovery | Saves continuing state and applies explicit feedback updates | Persistent output-adapter/AdamW path with retention loss; broader structural-learning interfaces |

These are project-specific implementations and connections added to the selected language backbone. They use established building blocks including attention, recurrent networks, residual connections, LoRA, AdamW and experience replay. Their development attribution does not establish priority over all related research. The brain-inspired names describe functional abstractions; anatomical equivalence has not been established.

The detailed inputs, state transitions, readouts and source paths are in [ARCHITECTURE.md](docs/ARCHITECTURE.md). [runtime/source/model/](runtime/source/model/) exposes the 91 source files decoded from the model bundle. This bundle includes historical research interfaces as well as active modules; the architecture document identifies the selected path.

## How shared computation works

The Qwen text decoder processes the prompt. At four layer sites, a learned projection maps its hidden state into the cognitive core. The core combines working state, retrieved experience, goals and candidates; a learned return path influences subsequent decoder computation. Language and judgment share these representations and experience. In the selected shared-choice configuration, public candidates use label-token logits from the same final core-modulated language state, read directly without parsing generated answers. Internal alternatives also have a native decision-field readout.

Spark has two implementation levels. A neural recurrent population generates and selects internal alternatives. The explicit lifecycle uses the same language model to propose textual descriptions, predictions and tests, encodes those exact descriptions for native ranking, stores them as hypotheses, and links later revisions to prior records. Python lifecycle code organizes these calls. Candidate-normalized scores and self-reported confidence are recorded with their respective meanings; they are not general calibrated truth probabilities.

The host performs file/network operations and supplies computation opportunities. Neural goal/action selection, operator execution, archival storage and host actions have separate responsibilities. Model-embedded weights and source still require a runtime that implements their execution.

## Files and execution formats

| Location | Contents | Execution |
|---|---|---|
| `Q8_0/` | 4,534,809,504-byte GGUF; Q8_0 language weights with cognitive tensors retained at required precision | Supplied Treyrin llama.cpp runtime |
| `BF16/` | 8,476,799,904-byte native GGUF; 9,705,116,525-byte mixed BF16/FP32 Transformers checkpoint; config, tokenizer and custom model code | Native runtime or Transformers custom-code loader |
| `runtime/` | Install/start/stop scripts, pinned architecture patch, exporter sources and unpacked model source | Runtime binaries/dependencies are a versioned Release asset |
| `docs/` | Architecture, objectives and checkpoint-specific results | Documentation |
| `licenses/` | Inherited Qwen license and attribution explanation | License records |

Q8_0 is quantized **Treyrin**, not the separate unadapted Qwen GGUF. Its metadata identifier is `treyrin`; the unadapted Qwen3.5 artifact uses `qwen35`. Quantization changes numerical precision and can change text and ranking behavior.

The six BF16 Python files implement the Transformers loader, model class, configuration, bundled architecture and state restoration. GGUF embeds the structural source, cognitive parameters, ONNX programs and initial state, so it does not need those neighboring Python files. Its native runtime still embeds Python for lifecycle operations. The language decoder executes through GGML/CUDA; cognitive ONNX and lifecycle/PyTorch work currently execute on CPU within the server process. The shipped runtime is therefore part of the execution requirement.

## Training lineage and measured scope

The selected `v4-mix-f` combination uses the core, bridges and language LoRA from step 40, with the output adapter from step 390. V5 assembly integrates the delivered parameters and state. Packaging and runtime repair add zero training steps. Earlier Qwen3-4B experiments and Ouro/SanSi evaluations belong to development history, rather than the current checkpoint's parameter source.

| Measurement | Recorded result | Interpretation |
|---|---|---|
| Selected v4 paired language check, 118 held-out items | Base 39; Treyrin 39; 2 wins, 2 losses, 114 ties | Similar totals in that task set |
| Strict multi-domain check, 25 items | Base 2; Treyrin 2 | Low exact-format accuracy in this small suite |
| Historical memory archive | 1,036,211 source tokens; 2,213 records | Archived history, not a simultaneously resident million-token KV window |
| Supported historical recall/revision/aggregation | 10/10 | Queries supported by the recorded source events |
| Selected v4 Spark run, 12 scenarios | Structural rubric 8.667/10; parsing/revision 12/12 | Automated structural/usefulness assessment, not a scientific-discovery success rate |
| Clean official llama.cpp b12000 loading test | Architecture error, exit code 1 | Unmodified upstream compatibility remains incomplete |

Checkpoint identities, intermediate regressions, exclusions and test conditions are in [RESULTS.md](docs/RESULTS.md). These measurements support runnable shared computation, selected task behavior, persistence and retrieval-grounded answers. Broad reasoning gains, durable beneficial individual learning and general decision quality require wider evidence.

## Development sources

The pretrained text decoder and tokenizer come from [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), inherited asset commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Native execution extends [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp); the patch pins upstream commit `48d22e295e2b86b47366c16390794f3e05ba970a`. PyTorch, Transformers and ONNX Runtime support definition, training, loading and execution.

Development also examined or evaluated [Ouro/SanSi](https://github.com/minnesotanlp/Sansi), [Jeff/Inner Jev](https://github.com/j-geuter/Jeff), [Laya](https://github.com/NandhaKishorM/laya), [CLM](https://github.com/Contrastive-LM/CLM), [Attentive Neural Processes](https://github.com/google-deepmind/neural-processes) and [CPA](https://github.com/zetayue/CPA). Hy-Memory, DeepInfer and Engram were considered in design discussions. These relationships describe sources and development history, not a competitive ranking or a claim that their weights are present in this checkpoint.

## License

Treyrin contributions use [Apache-2.0](LICENSE). [NOTICE](NOTICE) distinguishes the Treyrin changes from inherited Qwen assets. The original Qwen license is preserved in [licenses/QWEN-LICENSE](licenses/QWEN-LICENSE). Runtime dependencies retain their own distributed licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the runtime's license files.
