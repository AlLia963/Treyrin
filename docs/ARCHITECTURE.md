# Architecture, state transitions and implementation provenance

This document follows the v5 configuration and active call paths. The 91 bundled source files include both the selected architecture and historical research interfaces. File presence alone does not establish execution or measured benefit.

## 1. Decoder/core coupling

The language body is the Qwen3.5-4B text decoder and tokenizer, retaining its interleaved linear- and full-attention layers. The vision encoder is excluded. Core visits occur after zero-based layers 7, 15, 23 and 31.

At each visit, `to_core` projects language hidden states into width 32. The core combines working state, experience, candidates and goals; the return projection feeds its result into the language residual stream. Subsequent decoder layers process that result. LoRA and the output adapter change parameterized language behavior; core visits add a distinct forward-computation path.

The selected configuration uses an event clock and response cognitive stride 4. Temporary decoding caches bind to the current computation/request, while committed experience persists separately. The native server rebuilds derived context at a new turn when the cognitive boundary changes.

Source: [structural_model.py](../runtime/source/model/treyrin/structural_model.py), [qwen_interleaved.py](../runtime/source/model/treyrin/qwen_interleaved.py), [native patch](../runtime/source/treyrin.patch).

## 2. Regional recurrence and modulation

[regional_circuit.py](../runtime/source/model/treyrin/regional_circuit.py) defines explicit recurrent population states. Cortical IT/PT streams separate association and output flow. Selection populations include direct, indirect, external pallidal, subthalamic and output-inhibition states. Relay, pulvinar and reticular states participate in thalamic gating. Trainable rates mix old states with current proposals; some connections use softplus parameterization to constrain their sign.

Hippocampal context can bias competing proposals, selection outputs influence thalamic gating, and the resulting content affects later updates. These named compartments and connections are actual computations in the implementation. Their dimensions, connectivity and regional interpretation are engineering hypotheses, rather than established anatomical equivalence.

[r8_circuits.py](../runtime/source/model/treyrin/r8_circuits.py) adds prediction, valence, modulation and drives. Cerebellar-style next-observation prediction supplies an error signal; DA/ACh/NE states modulate writing, consolidation and reading; valence and accumulated drives influence goal/initiative-related computation. The utility of each pathway requires checkpoint-specific intervention evidence.

## 3. Experience admission, retrieval and consolidation

[structural_core.py](../runtime/source/model/treyrin/structural_core.py) manages shared experience. The current configuration has 8 hot episodic slots, 8192 archive slots, 4 semantic slots and 127 echo slots. Episodic keys, values, sources, occupancy and event identity participate in retrieval and updates. Eviction, replay and consolidation transfer information between levels; semantic slow state and associative matrices provide compressed representations alongside exact source records.

[episodic_policy.py](../runtime/source/model/treyrin/episodic_policy.py) consumes the current item, provenance, working context and existing slots, and chooses skip or a physical write slot. Execution is discrete; training uses a straight-through approximation. Zero-initialized residuals inherit an oldest-item baseline, while learned parameters can change admission and replacement. Future usefulness is a possible training target, not a future observation available during inference.

Retrieval combines query/source/key representations with competitive normalization, per-item gating, goal-dependent gain and an abstention exit. [hippocampal_circuit.py](../runtime/source/model/treyrin/hippocampal_circuit.py) provides separation, associative retrieval and predictive subcomputations. [code_attention.py](../runtime/source/model/treyrin/code_attention.py) performs multi-hop aggregation over token codes.

Neural slots are working/caching resources. Complete sources also enter durable journals and paged stores. Long records span pages; page size is not a total-record truncation boundary. Python payload-page defaults and native token-page implementations differ; the historical million-token check used 256-token pages. Enlarging these stores is distinct from enlarging the language KV window.

Source: [paging_store.py](../runtime/source/model/treyrin/paging_store.py), [live_memory.py](../runtime/source/model/live_memory.py), and native indexing/restoration in the patch.

## 4. Facts, internal content and revisions

Source records retain role, identity, time, epistemic type and revision relationships. Observed facts, generated hypotheses, intentions and goals have distinct records. Internal proposals enter the corresponding mental archive. Corrections and retractions keep links to preceding content, supporting historical interpretation beyond text similarity.

[cognitive_workspace.py](../runtime/source/model/treyrin/cognitive_workspace.py), [mental_archive.py](../runtime/source/model/treyrin/mental_archive.py) and [goal_state.py](../runtime/source/model/treyrin/goal_state.py) implement typed identities and transitions. Text and metadata support exact recovery and provenance; neural representations participate in forward computation. These contracts address source confusion, hypothesis/fact mixing and revision continuity. Factual accuracy also depends on the observations, retrieval, inference and learning quality.

## 5. Native judgments and Spark

[decision_field.py](../runtime/source/model/treyrin/decision_field.py) reads shared representations to score candidate descriptions. Descriptions can be encoded by the same language body. Scores normalize within the supplied or internally constructed candidate population.

The selected configuration also enables `shared_choice_readout`. When public choice-label binding is active, closed-candidate logits are read from label-token projections of the final core-modulated language hidden state. The native deployment uses that shared label path for supported public alphabets, while internal hypothesis populations use the native field. These are direct numerical readouts; the implementation does not infer the chosen option by parsing a generated answer string. The public-candidate path therefore should not be described as an independent, unrelated classifier.

[hypothesis_population.py](../runtime/source/model/treyrin/hypothesis_population.py) implements recurrent, input-conditioned alternatives with current capacity 4. A shared generator combines input, goals, working state, retrieved experience and prior alternatives. Learned gates determine occupancy; a selected survivor prevents an empty population. Allocation scores are competition signals whose semantics depend on training.

Spark targets useful, revisable explanations and directions under incomplete evidence. The explicit lifecycle in [spark_lifecycle.py](../runtime/source/model/spark_lifecycle.py) proceeds through:

1. Same-body generation of descriptions, predictions, tests, evidence identities and self-reported confidence.
2. Same-model encoding of those exact descriptions and native ranking.
3. Storage as typed hypotheses with provenance.
4. Later revision with new identities linked to preceding hypotheses.

The neural recurrent population and Python-organized textual proposal/ranking/revision calls are separate implementation levels. Confidence fields are marked uncalibrated. Structural validation checks fields and source references, rather than the scientific truth of a claim.

## 6. Joint goal/action selection

[goal_control.py](../runtime/source/model/treyrin/goal_control.py) normalizes once over goal/action pairs. Inputs include 29 per-goal features, 9 context features and a same-body 32-dimensional goal representation. Actions are `wait`, `think`, `speak`, `request_evidence` and `act`, with a synthetic no-goal row.

The legality mask uses occupancy, permission and resource facts. Deadlines, priorities and attempt histories enter preference computation, rather than a mask that directly prescribes speaking at a deadline. A host supplies execution opportunities, executes selected actions and returns observations. The current implementation contains initiative-related states, neural selection and interfaces; sustained useful autonomy requires longitudinal task evidence.

## 7. Individual learning and persistence

Offline interfaces support core, bridge, LoRA and output-adapter updates. The selected `v4-mix-f` lineage combines step-40 core/bridge/LoRA with the step-390 output adapter. V5 packaging and runtime repair add no training steps.

The deployed Q8_0 path persists experience, hypotheses/goals, individual output-adapter updates and optimizer moments. Explicit human-feedback learning applies AdamW with retention loss to that output adapter. Broader joint structural-learning interfaces are also bundled; their available update scope differs from the deployed native path.

Source: [individual_learning.py](../runtime/source/model/individual_learning.py), [output_adaptation.py](../runtime/source/model/treyrin/output_adaptation.py), [structural_learning.py](../runtime/source/model/treyrin/structural_learning.py). Model/state bindings include configuration, key structural source and parameter content, providing an explicit compatibility check during restoration.

## 8. Attribution and selected versus historical paths

Project-specific additions include the four core visits and return connections, regional state/connection design, episodic admission policy, typed mental/revision contracts, recurrent candidates and same-body Spark lifecycle, joint goal/action readout, and shared save/restore protocol. This identifies changes relative to the adopted backbone and earlier project implementations.

Attention, GRU, residual connections, LoRA, AdamW, replay and brain-inspired organization have existing research histories. Treyrin adapts and combines them for these objectives. Development attribution is separate from a field-wide novelty or priority determination.

Early Qwen3-4B decision experiments, Ouro/SanSi evaluation and RMT-style recurrent-state controls remain in research history or bundled source. Current weights descend from Qwen3.5-4B plus the selected Treyrin parameters. Backbone adapters and migration contracts exist; demonstrated transfer of learned cognition and old experiences to a new backbone remains future work.
