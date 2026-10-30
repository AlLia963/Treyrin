"""Standard TreyrinForCausalLM: one class that owns the real Treyrin graph.

R1 of order 000064.  The class is a real nn.Module / PreTrainedModel whose
direct children are the *actual* modules of the inherited candidate:

    body    HFBodyPort(NativeBodyPort) -> peft LoRA -> Qwen3.5-4B text decoder
            (+ output_adapter: readout-only adaptation)
    core    StructuralCore (cortex/thalamus/hippocampus/workspace/drives/...)
    to_core / to_body / feedback_gain   the bidirectional bridge

named_parameters() and state_dict() therefore expose the real tensors under
exactly the names the inherited candidate saved (body.*, core.*, to_core.*,
to_body.*, feedback_gain).  The graph object that implements the
forward/observation/learning contract (model.graph) shares these very module
objects, so the inherited binding and learning-contract identities stay
reproducible on the execution path that produced them.

Loading is strict: see treyrin_assembly.  There is no random-initialisation
path and no fallback to a baseline/expert/answer-cache path.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from treyrin_paths import MODEL_DIR, WORKSPACE, bootstrap

bootstrap()

from configuration_treyrin import TreyrinConfig
from treyrin_assembly import (
    TreyrinAssembly, TreyrinStrictLoadError, base_body_index, key_report, open_model,
    save_standard_package, trainable_parameters,
)
from treyrin_recipe import PROBES


class TreyrinForCausalLM(PreTrainedModel):
    """The standard entry point of the complete Treyrin model."""

    config_class = TreyrinConfig
    main_input_name = "input_ids"
    base_model_prefix = "treyrin"
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, config: TreyrinConfig, *, assembly: TreyrinAssembly | None = None,
                 **kwargs: Any) -> None:
        super().__init__(config)
        if assembly is None:
            raise TreyrinStrictLoadError(
                "TreyrinForCausalLM is loaded, never randomly initialised: use "
                "TreyrinForCausalLM.from_pretrained(config_dir_or_model_ready)")
        self._attach(assembly)

    # -- construction -------------------------------------------------------
    def _attach(self, assembly: TreyrinAssembly) -> None:
        graph = assembly.graph
        # The graph object is kept out of the module registry on purpose: its
        # children are registered *here*, and registering the graph itself
        # would duplicate every key.
        object.__setattr__(self, "_graph", graph)
        self.body = graph.body
        self.core = graph.core
        self.to_core = graph.to_core
        self.to_body = graph.to_body
        self.feedback_gain = graph.feedback_gain
        self.tokenizer = assembly.tokenizer
        self.resident_state = assembly.resident_state
        self.retention_reference = assembly.retention_reference
        self.base_body_dir = Path(assembly.base_body_dir)
        self.source = assembly.source
        self.load_report = assembly.load_report
        self._execution_state = None
        object.__setattr__(self, "_assembly", assembly)
        self._verify_module_ownership()

    def _verify_module_ownership(self) -> None:
        """Prove the class owns the same objects/keys as the executing graph."""
        graph = self._graph
        for name in ("body", "core", "to_core", "to_body"):
            if getattr(self, name) is not getattr(graph, name):
                raise TreyrinStrictLoadError(f"module {name} is not the inherited graph module")
        if self.feedback_gain is not graph.feedback_gain:
            raise TreyrinStrictLoadError("feedback_gain is not the inherited graph parameter")
        names_here = [name for name, _ in self.named_parameters()]
        names_graph = [name for name, _ in graph.named_parameters()]
        if names_here != names_graph:
            raise TreyrinStrictLoadError("named_parameters layout differs from the graph")
        if list(self.state_dict()) != list(graph.state_dict()):
            raise TreyrinStrictLoadError("state_dict layout differs from the graph")

    def post_init(self):
        raise TreyrinStrictLoadError(
            "post_init would re-initialise 4.2B verified parameters; TreyrinForCausalLM "
            "loads weights only from the verified assets/model-ready payload")

    # -- identity -----------------------------------------------------------
    @property
    def graph(self):
        """The inherited TreyrinStructuralModel that executes this model."""
        return self._graph

    @property
    def binding(self) -> str:
        return self._graph._binding

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def seal_parameters(self) -> str:
        binding = self._graph.seal_parameters()
        self.load_report["binding"] = binding
        self.load_report["binding_preserved"] = binding == self.load_report["binding_expected"]
        return binding

    def make_optimizer(self, group_names=None, *, weight_decay: float = 0.01,
                       body_lr: float = 1e-5, core_lr: float = 1e-4):
        """Build the optimizer on the *actual* trainable tensors.

        group_names restores the exact inherited grouping (the saved learning
        checkpoint requires the same partition); without it the inherited
        two-group body/core split is used.  This never performs an update.
        """
        trainable = trainable_parameters(self._graph)
        if group_names is not None:
            flat = [name for group in group_names for name in group]
            if len(set(flat)) != len(flat) or set(flat) != set(trainable):
                raise TreyrinStrictLoadError(
                    "saved optimizer groups do not cover the exact trainable parameter set")
            groups = [{"params": [trainable[name] for name in group]} for group in group_names]
        else:
            body = [p for name, p in trainable.items() if name.startswith("body.")]
            core = [p for name, p in trainable.items() if not name.startswith("body.")]
            groups = [{"params": body, "lr": body_lr}, {"params": core, "lr": core_lr}]
        return torch.optim.AdamW(groups, weight_decay=weight_decay)

    def parameter_groups(self) -> dict:
        """The exact parameter collection an optimizer must cover (body / core)."""
        trainable = trainable_parameters(self._graph)
        body = sorted(name for name in trainable if name.startswith("body."))
        core = sorted(name for name in trainable if not name.startswith("body."))
        return {"body": body, "core": core, "all": sorted(trainable)}

    def state_report(self) -> dict:
        return key_report(self._graph, self.resident_state,
                          base_layout=base_body_index(self.base_body_dir))

    def describe(self) -> dict:
        """Honest summary of what is actually loaded and executed."""
        parameters = list(self.named_parameters())
        trainable = trainable_parameters(self._graph)
        return {
            "schema": "treyrin-standard-model-description-v1",
            "state_schema": self.config.state_schema,
            "binding": self.binding,
            "inherited_binding": self.load_report["inherited"]["parameter_binding"],
            "inherited_binding_preserved": self.load_report["inherited"]["binding_preserved"],
            "inherited_contract_identity": self.load_report["inherited"]["contract_identity"],
            "named_parameters": len(parameters),
            "named_parameter_elements": int(sum(p.numel() for _, p in parameters)),
            "trainable_tensors": len(trainable),
            "trainable_elements": int(sum(p.numel() for p in trainable.values())),
            "state_dict_keys": len(self.state_dict()),
            "resident_state_tensors": len(self.resident_state.tensors()),
            "modules": {
                "language_body": type(self.body.model).__name__,
                "language_decoder": type(self.body.decoder).__name__,
                "lora_trainable": self.load_report["language_lora"]["trainable_tensors"],
                "output_adapter": type(self.body.output_adapter).__name__,
                "core": type(self.core).__name__,
                "bridge": ["to_core", "to_body", "feedback_gain"],
            },
            "load_context": self.load_report["context"],
            "optimizer_steps": self.load_report["inherited"]["optimizer_steps"],
            "capability_accepted": False,
            "formal_release": False,
            "not_yet_verified": [
                "joint multi-domain training (R3)",
                "one real common-graph gradient batch (R3)",
                "capability comparison against the base model (R6)",
            ],
        }

    # -- state --------------------------------------------------------------
    def _resolve_state(self, state, batch: int):
        if batch != 1:
            raise TreyrinStrictLoadError(
                "the inherited event clock executes one conversation; batch must be 1")
        if state is not None:
            return state
        if self._execution_state is None:
            loaded = self.resident_state
            if loaded.binding != self.binding:
                rebound = dataclasses.replace(loaded, binding=self.binding)
                self._graph.core.validate_state(rebound, 1)
                self.load_report["execution_state_rebound"] = {
                    "from": loaded.binding, "to": self.binding,
                    "reason": self.load_report.get("binding_note"),
                    "tensors_unchanged": True,
                }
                self._execution_state = rebound
            else:
                self._execution_state = loaded
        return self._execution_state

    # -- standard forward ---------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, labels=None, *,
                state=None, decision_positions=None, return_structural=False,
                **graph_options):
        """Read-only forward over the complete Treyrin graph.

        No observation/consolidation is committed here: the returned state must
        not be written back, so checkpoint recomputation stays side-effect free.
        """
        if input_ids is None:
            raise ValueError("input_ids required")
        ids = input_ids.long().to(self.device)
        mask = (torch.ones_like(ids) if attention_mask is None
                else attention_mask.to(self.device))
        state = self._resolve_state(state, ids.shape[0])
        output = self._graph(ids, mask, state, decision_positions=decision_positions,
                             **graph_options)
        if return_structural:
            return output
        logits = output.language_logits
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      hidden_states=(output.body_hidden,))

    def forward_structural(self, input_ids, attention_mask=None, *, state=None, **graph_options):
        """The raw StructuralOutput (language/native/control/read surfaces)."""
        ids = input_ids.long().to(self.device)
        mask = (torch.ones_like(ids) if attention_mask is None
                else attention_mask.to(self.device))
        state = self._resolve_state(state, ids.shape[0])
        return self._graph(ids, mask, state, **graph_options)

    # -- generation ---------------------------------------------------------
    def eos_token_ids(self) -> tuple[int, ...]:
        values = []
        candidates = [getattr(self.config, "eos_token_id", None)]
        body_config = getattr(getattr(self.body, "model", None), "config", None)
        if body_config is not None:
            candidates.append(getattr(body_config, "eos_token_id", None))
        for candidate in candidates:
            if candidate is None:
                continue
            values.extend(candidate if isinstance(candidate, (list, tuple, set)) else [candidate])
        return tuple(sorted({int(value) for value in values}))

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, *, max_new_tokens=None,
                 eos_token_id=None, state=None, **graph_options):
        """Real generation through the graph (real EOS/EOT, no fixed 64-token cap)."""
        if input_ids is None:
            raise ValueError("input_ids required")
        if self.training:
            self.eval()
        ids = input_ids.long().to(self.device)
        state = self._resolve_state(state, ids.shape[0])
        if eos_token_id is None:
            eos_token_id = self.eos_token_ids()
        result = self._graph.generate(ids, state, max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id or None, **graph_options)
        self.last_generation = getattr(self._graph, "last_generation", None)
        return result

    def learn_preference(self, question, preferred, *, steps=1, learning_rate=1e-3):
        """Apply an explicit individual readout update to the output adapter.

        Only the output adapter is optimized. The frozen body, graph, history
        and resident store are untouched; after the update the graph is resealed
        and the resident state is rebound to the new binding.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("nonempty question required")
        if not isinstance(preferred, str) or not preferred.strip():
            raise ValueError("nonempty preferred answer required")
        if type(steps) is not int or not 1 <= steps <= 500:
            raise ValueError("steps must be an integer in 1..500")
        adapter = getattr(getattr(self.graph, "body", None), "output_adapter", None)
        if adapter is None:
            raise ValueError("this package has no individual output adapter")
        tokenizer = self.tokenizer
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(preferred, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [int(self.eos_token_ids()[0])]
        ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=self.device)
        structural = self.forward_structural(ids)
        hidden = structural.body_hidden.detach()
        base = self.body.model.get_output_embeddings()(hidden).detach().float()
        parameters = list(adapter.parameters())
        for parameter in parameters:
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate))
        labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], dtype=torch.long,
                              device=self.device)
        before = None
        last = None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = base + adapter.delta(hidden).float()
            loss = torch.nn.functional.cross_entropy(
                logits[:, len(prompt_ids) - 1:-1].reshape(-1, logits.shape[-1]),
                labels[:, len(prompt_ids):].reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite individual loss; adapter unchanged")
            if before is None:
                before = float(loss.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            last = float(loss.detach())
        binding = self.graph.seal_parameters(force_rehash=True)
        self.resident_state = self.graph.initial_state(1)
        self.resident_state.binding = binding
        self._execution_state = None
        return dict(steps=steps, loss_before=before, loss_after=last,
                    adapter_tensors=len(parameters), binding=binding,
                    scope="output_adapter_only")

    def save_individual(self, path):
        """Persist only the individual output-adapter delta."""
        from safetensors.torch import save_file
        adapter = getattr(getattr(self.graph, "body", None), "output_adapter", None)
        if adapter is None:
            raise ValueError("no individual output adapter")
        tensors = {name: value.detach().cpu().contiguous()
                   for name, value in adapter.state_dict().items()}
        save_file(tensors, str(path))
        return dict(path=str(path), tensors=len(tensors),
                    adapter_rank=int(adapter.rank))

    def load_individual(self, path):
        """Load an individual adapter delta and reseal the resident graph."""
        from safetensors.torch import load_file
        adapter = getattr(getattr(self.graph, "body", None), "output_adapter", None)
        if adapter is None:
            raise ValueError("no individual output adapter")
        tensors = load_file(str(path))
        expected = adapter.state_dict()
        if set(tensors) != set(expected):
            raise ValueError("individual adapter keys differ from this package")
        for name, value in tensors.items():
            if tuple(value.shape) != tuple(expected[name].shape):
                raise ValueError("individual adapter shape differs: " + name)
            if not torch.isfinite(value).all():
                raise ValueError("individual adapter is nonfinite: " + name)
        adapter.load_state_dict(tensors, strict=True)
        for parameter in adapter.parameters():
            parameter.requires_grad = False
        binding = self.graph.seal_parameters(force_rehash=True)
        self.resident_state = self.graph.initial_state(1)
        self.resident_state.binding = binding
        self._execution_state = None
        return dict(path=str(path), tensors=len(tensors), binding=binding)

    # -- persistence --------------------------------------------------------
    def save_pretrained(self, save_directory, *, export_base: bool = False,
                        note: str | None = None, **kwargs) -> dict:
        """Write the verified trainable/state package (strictly reloadable)."""
        return save_standard_package(self, save_directory, export_base=export_base, note=note)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Strict load from a workspace config directory or a standard package.

        Accepted R1 options: device, dtype, quantize, verify_assets, strict,
        expect_inherited, require_binding, seal, tokenizer.  Unknown keyword
        arguments are recorded in the load report, never silently applied.
        """
        path = Path(pretrained_model_name_or_path)
        device = kwargs.pop("device", "cpu")
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))
        if dtype is not None and not isinstance(dtype, str):
            dtype = str(dtype).replace("torch.", "")
        quantize = kwargs.pop("quantize", None)
        verify_assets_mode = kwargs.pop("verify_assets", "full")
        strict = kwargs.pop("strict", True)
        expect_inherited = kwargs.pop("expect_inherited", True)
        require_binding = kwargs.pop("require_binding", None)
        seal = kwargs.pop("seal", True)
        tokenizer = kwargs.pop("tokenizer", None)
        retention_reference = kwargs.pop("retention_reference", False)
        ignored = dict(kwargs)
        assembly = open_model(
            path, device=device, dtype=dtype, quantize=quantize,
            verify_assets_mode=verify_assets_mode, strict_identity=strict,
            expect_inherited=expect_inherited, require_binding=require_binding,
            seal=seal, tokenizer=tokenizer, retention_reference=retention_reference)
        config = assembly.config
        config._name_or_path = str(path)
        body_config = getattr(getattr(assembly.body, "model", None), "config", None)
        if getattr(config, "eos_token_id", None) is None and body_config is not None:
            config.eos_token_id = getattr(body_config, "eos_token_id", None)
        assembly.load_report["ignored_kwargs"] = sorted(ignored)
        model = cls(config, assembly=assembly)
        model.eval()
        return model

    # -- learning contract (R3 hook, unchanged inherited implementation) ----
    def make_session(self, *, mode=None, retention_reference=None, auto_trim=None):
        """Bind the inherited StructuralLearningSession to this exact graph.

        The session verifies its own learning-contract identity; on the NF4/CUDA
        path that identity must equal the inherited b4151558... value.  No
        update is performed here.
        """
        from treyrin.structural_learning import ComputationMode, StructuralLearningSession
        from treyrin_paths import tokenizer_identity

        tokenizer = self.tokenizer

        def encode(text):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError("nonempty text required")
            return ids

        if retention_reference is None:
            retention_reference = self.retention_reference
        if retention_reference is None:
            raise TreyrinStrictLoadError(
                "the learning contract needs the frozen pre-adapter reference: load with "
                "from_pretrained(..., retention_reference=True) or pass retention_reference=")
        encoding = (self.load_report.get("encoding_sha256")
                    or tokenizer_identity(self.base_body_dir.parent))
        return StructuralLearningSession(
            self._graph, encode, encoding_sha256=encoding,
            retention_reference=retention_reference, retention_texts=PROBES,
            auto_trim=auto_trim, mode=mode or ComputationMode(amygdala_mode="normal"))

    def session_identity(self, session) -> dict:
        return {
            "contract_identity": session.contract_identity,
            "inherited_contract_identity": self.load_report["inherited"]["contract_identity"],
            "preserved": session.contract_identity == self.load_report["inherited"]["contract_identity"],
            "parameter_binding": session.state.binding,
        }
