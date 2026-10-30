---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
pipeline_tag: text-generation
language: [zh, en]
tags: [treyrin, gguf, custom_code, stateful]
---

# Treyrin v5

Treyrin is a stateful language-and-decision model implementation built from the Qwen3.5-4B text decoder and a learned recurrent cognitive core. Its development goal is one continuing individual whose experiences influence native judgments, language, hypotheses, goals and subsequent learning.

The implementation adds neural modules, parameters and state transitions to the decoder. It also includes programmatic memory and lifecycle procedures. This model card identifies their roles, checkpoint lineage, measured results and delivered scope.

## Files, precision and model identity

Two model directories contain three loading artifacts:

| Artifact | Location and size | Language precision | Execution |
|---|---|---|---|
| Transformers checkpoint | `BF16/model.safetensors`, 9.04 GiB | BF16 with FP32 cognitive/adaptation tensors | Treyrin Python implementation and Transformers |
| BF16 native model | `BF16/Treyrin-v5-BF16.gguf`, 7.89 GiB | BF16 | llama.cpp with Treyrin architecture support |
| Q8_0 native model | `Q8_0/Treyrin-v5-Q8_0.gguf`, 4.22 GiB | Q8_0 | The same Treyrin native implementation |

Q8_0 is the default native deployment. It is quantized **Treyrin**: its GGUF architecture identifier is `treyrin`. The separate, unadapted Qwen3.5-4B Q8_0 artifact has identifier `qwen35`.

The two Treyrin GGUFs contain byte-identical payloads for cognitive parameters, core inference, observation, consolidation, public-candidate inference, frozen core buffers, initial Spark state and structural source. Their language-weight representations differ. Quantization can change generated text and rankings while retaining the cognitive implementation.

### Why the BF16 directory contains Python

The BF16 directory combines the Transformers and native formats. Its six Python files serve the Transformers path:

| File | Role |
|---|---|
| `configuration_treyrin.py` | Treyrin configuration and architecture identifier |
| `modeling_treyrin.py` | `TreyrinForCausalLM`, including forward and generation interfaces |
| `modeling_treyrin_loader.py` | Graph construction, complete weight/state loading, and checkpoint saving |
| `modeling_treyrin_state.py` | Experience and Spark state registered in the model state dictionary |
| `modeling_treyrin_runtime.py` | A bundle of 91 architecture/lifecycle source modules, expanded into a runtime cache |
| `treyrin_transformers.py` | Package import entry point |

The code defines how model parameters and states participate in computation. Parameter values live in the checkpoint. Several bundled modules also implement multi-stage lifecycle procedures, described below.

Q8_0 contains only the native format. Its source and ONNX programs are embedded in GGUF metadata, so separate Python files are absent from that directory. Native lifecycle operations still use an embedded Python interpreter. The BF16 GGUF uses the same arrangement and runs independently of neighboring Transformers Python files. The difference follows the loading formats, rather than BF16 versus 8-bit arithmetic.

Treyrin currently uses a custom-code Transformers release. A framework-integrated release, with the architecture supplied by the installed framework and only model assets in the model directory, remains an integration task. Hugging Face documents [custom model implementation and distribution](https://huggingface.co/docs/transformers/main/en/custom_models).

## Implemented architecture

### Language and cognition share a forward path

The 32-layer, 2,560-wide Qwen3.5 text decoder exchanges representations with a 32-wide recurrent core at layers **7, 15, 23 and 31**. Learned projections map decoder representations into the core and return cognitive feedback to later decoder computation. Native judgment and language use this shared computation and experience state.

The inherited trainable additions comprise **501 tensors and 3,447,187 elements**, including language LoRA, output adaptation, cognitive parameters and decoder bridges. The Transformers state dictionary has **1,014 tensors**: 933 graph tensors, 76 experience-state tensors and 5 Spark-state tensors. Its 260 `core.*` entries include 256 parameter entries and four frozen buffers.

The ordinary native GGUF tensor section has 426 language tensors. Cognitive/adaptation parameters are also carried in embedded payloads. Counts across these sections require accounting for LoRA aliases and tied weights.

| Component | Concrete computation or state |
|---|---|
| Regional recurrence | Cortex/work-state updates, interacting streams and shared cognitive representations |
| Routing and selection | Thalamic-style gates and basal-ganglia-style selection loops |
| Episodic and semantic memory | Event encoding, bounded neural banks, associative retrieval, archive admission and consolidation |
| Prediction and modulation | Cerebellar-style prediction, plasticity signals, valence, arousal and drive state |
| Cognitive workspace | Typed observations, hypotheses, goals, outputs and completion records with identities and revision relationships |
| Native judgment | Candidate readout using language representations, cognitive state and descriptions in a declared option space |

These names denote functional abstractions with implemented neural updates and connections. Biological correspondence is a research interpretation. Candidate scores are normalized over supplied options or an internal population; calibration and coverage of alternatives require task evidence.

### Continuing experience

Source records retain role, identity, time, epistemic type and revision relationships. A disk-backed token-page store retains longer history. Observation and consolidation operators update neural state. Retrieval combines learned state-dependent representations and token-address/content features. Selected pages enter the active context; a continuing cursor exposes further pages.

The physical decoder window is **262,144 tokens**. Million-token operation concerns a longer stored stream and answers over retrieved history. Bounded neural banks and the page store have complementary roles. Bank updates, consolidation and selection are implemented; robust task-sensitive retention and compression across varied long-running work remain capability questions.

### Spark

Spark combines an internal recurrent hypothesis population with explicit proposal records. The population has vectors, gates, selection logits and attention state. Explicit records contain explanations, predicted observations, proposed tests, evidence references and model-reported confidence.

The same decoder generates proposals, and native candidate computation ranks their descriptions. Typed admission retains proposals as hypotheses. Revisions create new identities linked to prior records, allowing a line of thought to continue as evidence changes.

Python lifecycle procedures currently organize explicit generation, ranking and revision through staged calls to the same decoder. The recurrent hypothesis network is neural; the explicit workflow also includes programmed control flow. The original Spark objective is useful intuition and new directions from incomplete information. Scenario scores measure portions of that objective.

### Goals and learning

Goal intake links goals to observed user messages. The joint goal/action policy uses goals, elapsed time, resources, interruption cost and deadlines. The host executes selected actions and returns results. An idle scheduler supplies computation opportunities; the neural policy selects actions at those opportunities.

Individual state persists neural experience, hypotheses/goals, output-adapter updates and optimizer moments. Native explicit human-feedback learning applies AdamW with a retention loss to the individual output adapter. Broader Python research interfaces support joint structural learning. The deployed Q8_0 path performs state updates and output-adapter learning; whole-core/decoder online learning belongs to the broader research implementation.

## Development and training lineage

Early work explored a Qwen3-4B decision readout and evaluated pretrained Ouro/SanSi weights. The official SanSi local diagnostic performed zero local training updates. Those baselines remain in development records. Current Treyrin loads the Qwen3.5-4B language body and Treyrin core/adaptation weights, rather than SanSi weights.

Later work introduced the interleaved core, role-aware experience, native judgments, shared learning interfaces, persistent sessions and goal control. A calibration branch collected 400 states and updated four calibration tensors. Other joint updates belong to their own checkpoints.

The v4 run reached step 390. The exported combination, **`v4-mix-f`**, uses core, bridges and language LoRA from **step 40**, plus the output adapter from **step 390**. V5 assembly retains that delivered parameter lineage and integrates experience/Spark state into a complete checkpoint. Assembly and native runtime repair add zero training steps.

An inherited optimizer counter of 2,351 and the v4 run's 390-step counter describe different lineage stages. A downloaded inventory of 3,856,882 raw records describes available data, rather than examples consumed in final training. The step-390 exposure audit counted 88 distinct training addresses for its comparison.

## Recorded results

| Run/checkpoint | Measurement | Result | Scope |
|---|---|---|---|
| Historical v15c/v15d execution | Eight sandbox scenarios | 31/31 ready targets | Restricted file-task execution |
| Historical v15d lifecycle | Fresh-process recovery | 7/7 checks | Session, role and state recovery |
| Historical v3, `V6-RESULTS` | 235 strict development/regression items | Base 181; Treyrin 182 | One-item difference on that set |
| v4 step 80 / 190 / 290 / 390 | 118 core held-out items | Base 39 throughout; Treyrin 37 / 38 / 39 / 39 | Intermediate regressions and later recovery |
| `v4-mix-f`, `r6-mixf-full` | 118 core held-out items | Base 39; Treyrin 39; 2 wins, 2 losses, 114 ties | 96 mathematics and 22 label items |
| `v4-mix-f`, `r6-mixf-multi-strict` | 25 strict multi-domain items | Base 2; Treyrin 2 | Low exact-format accuracy in this small suite |
| Historical v4 memory store | Source history | 1,036,211 measured tokens; 2,213 records | 256-token pages; 2,200 records span pages |
| Historical v4 memory-forward run | Supported recall/revision/aggregation | 10/10 | One unsupported retraction expectation excluded because its source event was absent |
| Historical v3 Spark checkpoint | 12 scenarios | Structural rubric 8.042/10; recorded agent audit 7.583/10 | Natural/final cap hits 8/12 and 9/12; unequal-limit S12 rerun |
| Selected v4 Spark run | 12 scenarios | Structural rubric 8.667/10; 11/12 at least 7; structured/revision parsing 12/12 | No fixed answer cap; automated structural/usefulness rubric |
| Native runtime repair | Independent-directory Q8_0 startup, reply, lifecycle and archival checks | Passed | Integration and source-preservation checks |

R6 disables memory replay, correction routing and candidate readout in paired language comparisons. Those scores characterize the tested weight paths rather than isolating each cognitive mechanism's effect. The development set, held-out sets and Spark rubric differ in task and checkpoint conditions.

The evidence establishes runnable shared computation, selected task behavior, persistence and retrieval-grounded answers. Broad reasoning improvement, mature general decision competence, durable beneficial individual learning and scientific-discovery success require broader evidence than these measurements provide.

## Original objectives and delivered scope

| Objective | Delivered implementation | Remaining scope |
|---|---|---|
| Shared language and native decision model | Shared forward path, judgments, generation and experience state | Stable mutual benefit between judgment and open language |
| Functional brain-inspired structure | Registered recurrence, routing, prediction, modulation and memory connections | Checkpoint-aligned causal evidence for each mechanism's useful role |
| Intuition and Spark | Recurrent population, proposals, ranking, hypothesis storage and revision | Reliable useful directions with incomplete information; deeper workflow internalization |
| Continuing single conversation | Persistent identity/state, journal, page store and restoration | Sustained varied work and learning over long periods |
| Million-token useful history | Historical ingestion and retrieved-history answer results | Broad retention, correction, aggregation and compression behavior |
| Pretrained ability retention/enhancement | Paired comparisons and retention interfaces | Broad enhancement and retention across updates |
| Internally initiated behavior | Neural goal/action choice and idle-opportunity scheduling | Continual internally developing thought and robust spontaneous conversation |
| Individual adaptation | Persistent output-adapter/AdamW path and structural-learning interfaces | Durable personal gains with old knowledge and experiences retained |
| Future-backbone inheritance | Reusable interfaces and state/binding contracts | Demonstrated transfer of learned cognition and old experiences |
| Mainstream framework distribution | Complete custom-code checkpoint and native package | Framework-native registration and unmodified upstream llama compatibility |

## Development sources and implementation provenance

Treyrin's development uses pretrained assets, software libraries, evaluated models and methodological references. Their relationships to the delivered implementation differ:

| Source | Relationship to development and the release |
|---|---|
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Direct source of the pretrained text-decoder weights and tokenizer. Treyrin adds its cognitive modules, adaptation, state and decoder connections. |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | Direct native execution foundation. The release extends its model loading and computation paths for Treyrin and retains the source patch and upstream revision. |
| PyTorch, Transformers and ONNX Runtime | Frameworks used to define, train, load, export and execute the model. Their roles and licenses are recorded with the corresponding package. |
| [Ouro / SanSi](https://github.com/minnesotanlp/Sansi) | Early pretrained-model evaluation and a reference for looped typed decision readouts and learning interfaces. Local diagnostic records retain the evaluated official weights; the current checkpoint's parameter lineage is Qwen3.5-4B plus the Treyrin additions described above. |
| [Jeff / Inner Jev](https://github.com/j-geuter/Jeff), [Laya](https://github.com/NandhaKishorM/laya) and [CLM](https://github.com/Contrastive-LM/CLM) | Decision-model implementation references examined during development, including option encoding, direct readout, calibration and candidate representations. |
| [Attentive Neural Processes](https://github.com/google-deepmind/neural-processes) and [CPA](https://github.com/zetayue/CPA) | Source-level references examined for query-dependent aggregation and quantity-sensitive attention. Their examination is recorded in the early development review. |

Early design discussions also considered Hy-Memory, DeepInfer and Engram. Their role belongs to the project's design history. Component descriptions in this model card identify the actual Treyrin implementation and its measured behavior; dependency and weight attribution identify the assets used in the delivered checkpoint.

The resulting Treyrin implementation couples a persistent cognitive core to the decoder, maintains typed experiences and hypotheses, provides native judgments and open language, and carries individual state across saved sessions. These are concrete development outcomes. The result tables describe the behavior measured at the stated checkpoints; the objective table records the remaining work. The release is intended to make this computation and its state transitions available for inspection, continued development and research.

## Running the release

```powershell
.\Run-Llama.ps1 -Variant Q8_0
```

The launcher retrieves missing release assets and starts `http://127.0.0.1:18888`. Runtime files and individual state use external writable directories. BF16 is selected explicitly; `Download.ps1 -Variant Transformers` retrieves source-format weights. Transformers dependencies are in `BF16/requirements.txt`.

The native runtime targets **Windows x64 with a compatible NVIDIA driver**. It supplies llama.cpp with Treyrin support, CUDA, ONNX Runtime, and Python/CPU PyTorch. Language computation uses CUDA; cognitive/lifecycle work uses CPU in the same process. OpenAI Chat, OpenAI Responses and Anthropic Messages routes are implemented, with 14 optional MCP tools for web access and model lifecycle operations.

Unmodified llama.cpp b10360 does not recognize `treyrin`. A clean upstream b12000 test on 2026-10-30, using the matching official Windows CUDA release at commit `68d20b884c3878217de79eade0a249efde96ab38`, also failed to load this Q8_0 artifact with `unknown model architecture: 'treyrin'` (exit code 1). The supplied libraries include the architecture changes; the source patch and upstream revision accompany the runtime archive. Unmodified upstream compatibility remains incomplete.

## Origin and license

Language weights and tokenizer originate from **Qwen/Qwen3.5-4B**, Qwen team / Alibaba Cloud, commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The text decoder is used. Treyrin adds cognitive modules, decoder bridges, adaptation, state integration, hypothesis lifecycle and goal/action control.

Model and Treyrin source: [Apache-2.0](LICENSE). Attribution and dependency licensing: [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Runtime dependencies retain their distributed licenses.
