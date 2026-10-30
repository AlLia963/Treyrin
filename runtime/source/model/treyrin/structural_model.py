"""One language body and an internal recurrent core; no external decision API.

The body is reused with causally aligned internal feedback. Working scratch is
reset to the committed observation state on EACH body round to avoid leaking
future teacher tokens backward. Query output never overwrites experience.
"""
import hashlib
import inspect
import json
import weakref
from dataclasses import asdict, dataclass,replace
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from .structural_core import (CoreConfig, ExperienceState, StructuralCore,
                              causal_ngram_fingerprints)
from .interleaved_body import llama_interleaved
from .hypothesis_population import HypothesisPopulation
from .cognitive_workspace import CognitiveTick,KINDS,STATUSES,ACTIONS
from .goal_control import context_features,goal_feature_rows,legality_mask


class HFBodyPort(nn.Module):
    """Locally reviewed HF causal decoder port; class support must be tested.

    Reusing an HF model does not make the core model-independent in performance.
    No downloads, remote code, hooks, global layer mutation or active KV cache.
    """
    def __init__(self, causal_model, asset_identity: str,*,checkpoint_layers=False,choice_label_token_ids=None,output_adapter_rank=0):
        super().__init__()
        if not asset_identity or not hasattr(causal_model, 'base_model'):
            raise ValueError('Explicit local asset identity and a causal model required')
        self.model = causal_model
        self.asset_identity = asset_identity
        self.hidden_size = causal_model.config.hidden_size
        self.vocab_size = causal_model.config.vocab_size
        if type(output_adapter_rank)is not int or output_adapter_rank<0:raise ValueError('Explicit output-only adapter rank required')
        self.output_adapter=None
        if output_adapter_rank:
            from .output_adaptation import OutputAdapter
            self.output_adapter=OutputAdapter(self.hidden_size,self.vocab_size,output_adapter_rank,
                device=next(causal_model.parameters()).device)
        if type(checkpoint_layers) is not bool:raise ValueError('Explicit checkpoint policy required')
        self.checkpoint_layers=checkpoint_layers
        self.register_buffer('choice_label_tokens',None if choice_label_token_ids is None else
            torch.tensor(choice_label_token_ids,dtype=torch.long,device=next(causal_model.parameters()).device))
        # Source identity of the installed implementation is not a claim that
        # a matching config alone guarantees identical execution.
        source = Path(inspect.getfile(type(self.decoder)))
        self.runtime_identity = {'decoder_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'torch_version': torch.__version__,'pure_decoder_checkpointing':checkpoint_layers,
                                 'public_choice_label_ids':choice_label_token_ids}
        self.runtime_identity['output_only_adapter_rank']=output_adapter_rank
        self.runtime_identity['output_only_adapter_source_sha256']=hashlib.sha256(Path(__file__).with_name('output_adaptation.py').read_bytes()).hexdigest()

    @property
    def supports_choice_prior(self):return self.choice_label_tokens is not None

    def choice_label_logits(self,hidden,count):
        if not self.supports_choice_prior or not 1<=count<=len(self.choice_label_tokens):raise ValueError('Reviewed public label alphabet required')
        rows=self.choice_label_tokens[:count];head=self.model.get_output_embeddings()
        weight=head.weight.index_select(0,rows)
        bias=head.bias.index_select(0,rows)if getattr(head,'bias',None)is not None else None
        normalized=self.decoder.norm(hidden)
        base=torch.nn.functional.linear(normalized,weight,bias)
        if self.output_adapter is not None:base=base+self.output_adapter.label_delta(normalized,rows).to(base.dtype)
        return base.float()

    @property
    def decoder(self):
        # PEFT wraps a causal LM in a LoraModel. Its get_base_model() returns
        # the original causal LM with injected adapter modules still attached.
        causal = (self.model.get_base_model() if hasattr(self.model, 'peft_config')
                  else self.model)
        return causal.base_model

    def embed(self, ids):
        return self.model.get_input_embeddings()(ids)

    def hidden(self, embeddings, mask):
        if self.checkpoint_layers and torch.is_grad_enabled():
            return self._interleaved(embeddings, mask, None, (), checkpoint_layers=True)
        result = self.decoder(inputs_embeds=embeddings, attention_mask=mask,
            use_cache=False, return_dict=True)
        return result.last_hidden_state

    def logits(self, hidden):
        base=self.model.get_output_embeddings()(hidden)
        if self.output_adapter is not None:base=base+self.output_adapter.delta(hidden).to(base.dtype)
        return base

    def _interleaved(self, embeddings, mask, visit, sites, *, checkpoint_layers=False):
        from .interleaved_body import llama_interleaved
        from .qwen_interleaved import qwen_interleaved
        decoder = self.decoder
        name = type(decoder).__name__
        if name == 'LlamaModel':
            runner = llama_interleaved
        elif name == 'Qwen3_5TextModel':
            runner = qwen_interleaved
        else:
            raise ValueError(f'No reviewed layer-interleaved adapter for {name}')
        return runner(decoder, embeddings, mask, visit, sites,
                      checkpoint_layers=checkpoint_layers)

    def hidden_interleaved(self, embeddings, mask, visit, sites):
        return self._interleaved(embeddings, mask, visit, sites,
                                 checkpoint_layers=self.checkpoint_layers)

    def hidden_interleaved_cached(self,embeddings,mask,visit,sites,cache):
        from .qwen_interleaved import qwen_interleaved_cached
        return qwen_interleaved_cached(self.decoder,embeddings,mask,visit,sites,cache)


@dataclass
class StructuralOutput:
    language_logits: Tensor       # B,T,V; position i predicts token i+1
    native_logits: Tensor         # B,K; at explicit last-prompt positions
    control_logits: Tensor        # B,3; uncalibrated auxiliary output
    predicted_outcomes: Tensor    # B,K,D
    shared_hidden: Tensor         # B,T,D; final round, causal per position
    route_weights: Tensor         # B,T,6
    candidate_reads: Tensor       # B,T,K,E
    scratch_working: Tensor       # B,L,D; NOT committed experience
    predicate_logits: Tensor | None = None   # B,K; binary, not softmax classes
    action_values: Tensor | None = None      # B,K; utility, not belief
    read_allocation: Tensor | None = None    # B,T,K; actual last inner read
    scratch_regions: tuple | None = None
    regional_edges: Tensor | None = None
    hypothesis_vectors: Tensor | None = None
    hypothesis_gates: Tensor | None = None
    hypothesis_episode_attention: Tensor | None = None
    scratch_hippocampal: Tensor | None = None
    hippocampal_mismatch: Tensor | None = None
    # r8 per-step scratch surfaces
    thalamic: Tensor | None = None            # B,T,3,D
    cerebellar_prediction: Tensor | None = None  # B,T,D
    cerebellar_errors: Tensor | None = None   # B,T,D; self-supervised target surface
    dopamine: Tensor | None = None            # B,T
    neuromod: Tensor | None = None            # B,T,3
    drive_vector: Tensor | None = None        # B,T,D
    initiate_logits: Tensor | None = None     # B,T
    cognitive_controls: Tensor | None = None  # B,T,5; model-internal opportunity policy
    body_hidden: Tensor | None = None  # shared body/core perception, not only regional activity
    soft_candidate_reads: Tensor | None = None  # training scratch, never substitutes hard forward retrieval
    hypothesis_selection_logits: Tensor | None = None


class InferencePrefixCache:
    """Request-local exact cache. A byte-different causal prefix forces replay.

    Never stores across conversations, parameter changes, modes or experience.
    It does not approximate the recurrent scan or remove any cortical visits.
    """
    def __init__(self):
        self.context=None;self.scans={};self.descriptions={}
        self.reused_positions=0;self.executed_positions=0;self.prefix_misses=0

    def bind(self,context):
        if self.context!=context:
            self.context=context;self.scans.clear();self.descriptions.clear()


def _cache_tensor(tensor):
    return None if tensor is None else (id(tensor),tensor._version,tuple(tensor.shape),str(tensor.device),str(tensor.dtype))


class AutoregressiveDecodeCache(InferencePrefixCache):
    """Single request: native body KV/GDN plus per-site causal core scratch."""
    def __init__(self):
        super().__init__();self.body_cache=None;self.seen_tokens=0;self.tokens=None;self.invalid=False;self.cognitive_boundary=None;self.choice_prior=None

    def bind(self,context):
        if self.invalid:raise ValueError('Incomplete decoder scratch cannot be reused')
        if self.context is not None and self.context!=context:raise ValueError('Decoder scratch belongs to different experience/parameters/mode/candidates')
        super().bind(context)


class TreyrinStructuralModel(nn.Module):
    def __init__(self, body: nn.Module, config: CoreConfig, *, freeze_body: bool | None=True):
        super().__init__()
        self.body, self.config = body, config
        if freeze_body is not None and type(freeze_body) is not bool:
            raise ValueError('Use True/False or None to preserve an explicit body training mask')
        if freeze_body is not None:
            for p in body.parameters():
                if not freeze_body and not (p.is_floating_point() or p.is_complex()):
                    raise ValueError('Quantized weights cannot be unfrozen; use freeze_body=None to preserve the LoRA mask')
                p.requires_grad_(not freeze_body)
        self.core = StructuralCore(config)
        self.to_core = nn.Linear(body.hidden_size, config.width, bias=False)
        self.to_body = nn.Linear(config.width, body.hidden_size, bias=False)
        # Small, nonzero initialization permits writer/language gradients at start.
        nn.init.normal_(self.to_body.weight, std=0.001)
        self.feedback_gain = nn.Parameter(torch.tensor(0.02))
        # Keep cognitive parameters in fp32, on the language body's device.
        # Frozen quantized body parameters retain their original dtype/mask.
        device = next(body.parameters()).device
        self.core.to(device=device)
        self.to_core.to(device=device)
        self.to_body.to(device=device)
        self.feedback_gain = nn.Parameter(self.feedback_gain.detach().to(device))
        self._binding = ''
        self._sealed_versions = None
        self._sealed_configuration = None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._binding = ''
        self._sealed_versions = None
        self._sealed_configuration = None
        return result

    def _execution_configuration(self):
        native_body = getattr(self.body, 'model', self.body)
        config = getattr(native_body, 'config', None)
        return json.dumps(dict(core=asdict(self.config),asset=self.body.asset_identity,
            body_class=type(native_body).__module__+'.'+type(native_body).__qualname__,
            native_field_source=hashlib.sha256((Path(__file__).parent/'decision_field.py').read_bytes()).hexdigest(),
            episodic_policy_source=hashlib.sha256((Path(__file__).parent/'episodic_policy.py').read_bytes()).hexdigest(),
            cognitive_workspace_source=hashlib.sha256((Path(__file__).parent/'cognitive_workspace.py').read_bytes()).hexdigest(),
            interleaved_source=hashlib.sha256((Path(__file__).parent/'interleaved_body.py').read_bytes()).hexdigest(),
            qwen_source=hashlib.sha256((Path(__file__).parent/'qwen_interleaved.py').read_bytes()).hexdigest(),
            regional_source=hashlib.sha256((Path(__file__).parent/'regional_circuit.py').read_bytes()).hexdigest(),
            hypothesis_source=hashlib.sha256((Path(__file__).parent/'hypothesis_population.py').read_bytes()).hexdigest(),
            hippocampal_source=hashlib.sha256((Path(__file__).parent/'hippocampal_circuit.py').read_bytes()).hexdigest(),
            r8_source=hashlib.sha256((Path(__file__).parent/'r8_circuits.py').read_bytes()).hexdigest(),
            body_runtime=getattr(self.body, 'runtime_identity', None),
            trainable_body_names=[name for name,p in self.body.named_parameters() if p.requires_grad],
            body_configuration=config.to_dict() if config is not None else None),
            sort_keys=True,default=str)

    def seal_parameters(self,*,force_rehash=False) -> str:
        """Content identity; recompute after optimization BEFORE reencoding history.

        Parameter-version checks catch normal optimizer updates in-process.
        Bypassing PyTorch with .data writes is unsupported. Saved assets are
        additionally checked by whole-file hashes, not just a caller label.
        """
        h = hashlib.sha256()
        configuration = self._execution_configuration()
        h.update(configuration.encode())
        h.update(b'treyrin-structural-merkle-v1')
        for filename in ('structural_core.py','structural_model.py','decision_field.py','interleaved_body.py','regional_circuit.py','hypothesis_population.py','hippocampal_circuit.py','r8_circuits.py','qwen_interleaved.py','episodic_policy.py','cognitive_workspace.py'):
            h.update((Path(__file__).parent/filename).read_bytes())
        cache=getattr(self,'_tensor_digest_cache',{})
        next_cache={};hits=hashed=byte_count=0
        for name, tensor in self.state_dict(keep_vars=True).items():
            h.update(name.encode())
            h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            signature=(tensor._version,tuple(tensor.shape),str(tensor.dtype),str(tensor.device))
            previous=cache.get(name)
            if not force_rehash and previous is not None and previous[0]()is tensor and previous[1]==signature:
                digest=previous[2];hits+=1
            else:
                payload=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                digest=hashlib.sha256(payload).digest();hashed+=1;byte_count+=len(payload)
            next_cache[name]=(weakref.ref(tensor),signature,digest)
            h.update(digest)
        self._tensor_digest_cache=next_cache
        self._seal_statistics=dict(cache_hits=hits,tensors_hashed=hashed,bytes_hashed=byte_count,forced=force_rehash)
        self._binding = h.hexdigest()
        self._sealed_versions = self._version_stamp()
        self._sealed_configuration = configuration
        return self._binding

    def _version_stamp(self):
        # Connectivity buffers are part of the computation too. A mask edit
        # cannot silently keep an experience state bound to the former graph.
        return tuple((name,id(value),value._version) for name,value in
                     (*self.named_parameters(),*self.named_buffers()))

    def _check(self, state, batch):
        if not self._binding or self._sealed_versions != self._version_stamp():
            raise ValueError('Parameters changed/unsealed: seal and reencode observations')
        if self._sealed_configuration != self._execution_configuration():
            raise ValueError('Body configuration changed: seal and reencode observations')
        if state.binding != self._binding:
            raise ValueError('Experience belongs to different parameter content')
        self.core.validate_state(state, batch)

    def initial_state(self, batch):
        if not self._binding:
            self.seal_parameters()
        result = self.core.initial_state(batch, self._binding)
        self._check(result, batch)
        return result

    def _tokens(self, ids, mask):
        if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
            raise ValueError('Nonempty long token matrix required')
        if mask.shape != ids.shape or mask.device != ids.device or not ((mask==0)|(mask==1)).all():
            raise ValueError('Binary attention mask required')
        if not mask.bool().any(-1).all():
            raise ValueError('Empty batch row')
        if ids.min() < 0 or ids.max() >= self.body.vocab_size:
            raise ValueError('Token outside body vocabulary')
        if ids.device != next(self.parameters()).device:
            raise ValueError('Input device differs from model')

    def encode_descriptions(self, ids, mask):
        """Public description tokens B,K,L; not candidate IDs or teacher answers."""
        if ids.ndim != 3 or ids.dtype != torch.long or min(ids.shape) < 1:
            raise ValueError('Nonempty candidate/goal descriptions required')
        b,k,t = ids.shape
        flat_ids,flat_mask = ids.reshape(b*k,t),mask.reshape(b*k,t)
        self._tokens(flat_ids,flat_mask)
        if self.config.description_encoder == 'embedding_mean':
            # Historical bag-of-words control: almost order-invariant. Never
            # promote this mode as sufficient candidate/goal language semantics.
            embeddings = self.to_core(self.body.embed(ids).to(self.to_core.weight.dtype))
            return (embeddings*mask[:,:,:,None]).sum(2)/mask.sum(2)[:,:,None]
        hidden = self.body.hidden(self.body.embed(flat_ids),flat_mask)
        positions = torch.arange(t,device=ids.device)[None].expand(b*k,-1)
        last = positions.masked_fill(~flat_mask.bool(),-1).max(-1).values
        summary = hidden[torch.arange(b*k,device=ids.device),last]
        return self.to_core(summary.to(self.to_core.weight.dtype)).reshape(b,k,-1)

    def forward(self, ids, mask, state, *, decision_positions=None, candidate_ids=None,
                candidate_mask=None, goal_ids=None, goal_mask=None,
                read_mode='candidate', feedback_mode='normal', body_feedback=True,circuit_mode='normal',
                hippocampal_mode='normal',thalamic_mode='normal',cerebellar_mode='normal',
                amygdala_mode='off',drift_level=None,_inference_cache=None,_decode_cache=None,_cognitive_boundary=None,body_choice_labels=False,_last_token_logits=False,_hypothesis_state=None):
        self._tokens(ids, mask)
        b,t = ids.shape
        self._check(state, b)
        if type(body_choice_labels)is not bool:raise ValueError('Explicit public choice-label binding required')
        if type(_last_token_logits)is not bool:raise ValueError('Explicit vocabulary projection mode required')
        if _decode_cache is not None and _inference_cache is not None:raise ValueError('Choose one inference scratch strategy')
        cache=_decode_cache if _decode_cache is not None else _inference_cache
        incremental=isinstance(cache,AutoregressiveDecodeCache)
        if _decode_cache is not None and not incremental:raise ValueError('Explicit autoregressive cache required')
        if cache is not None:
            if not isinstance(cache,InferencePrefixCache) or torch.is_grad_enabled() or self.training:
                raise ValueError('Request-local prefix cache is inference-only')
            cache.bind((id(self),self._binding,id(state),tuple((n,_cache_tensor(v)) for n,v in state.tensors().items()),
                _cache_tensor(candidate_ids),_cache_tensor(candidate_mask),_cache_tensor(goal_ids),_cache_tensor(goal_mask),
                read_mode,feedback_mode,body_feedback,circuit_mode,hippocampal_mode,thalamic_mode,cerebellar_mode,amygdala_mode,
                _cache_tensor(drift_level),body_choice_labels))
        if incremental and (self.config.coupling_mode!='layer_interleaved' or not hasattr(self.body,'decoder') or
            type(self.body.decoder).__name__!='Qwen3_5TextModel' or b!=1 or not bool((mask==1).all())):
            raise ValueError('Incremental mode supports one unpadded reviewed Qwen interleaved stream')

        def descriptions(key,tokens,token_mask):
            if cache is not None and key in cache.descriptions:return cache.descriptions[key]
            result=self.encode_descriptions(tokens,token_mask)
            if cache is not None:cache.descriptions[key]=result
            return result
        if decision_positions is None:
            positions = torch.arange(t, device=ids.device)[None].expand(b,-1)
            decision_positions = positions.masked_fill(~mask.bool(),-1).max(-1).values
        if decision_positions.shape != (b,) or decision_positions.dtype != torch.long:
            raise ValueError('One explicit last-prompt position per row required')
        if (decision_positions<0).any() or (decision_positions>=t).any():
            raise ValueError('Decision position outside sequence')
        rows = torch.arange(b, device=ids.device)
        if not mask[rows,decision_positions].bool().all():
            raise ValueError('Decision cannot occur in padding')
        event_clock=self.config.cognitive_clock=='event'
        if event_clock and b!=1:raise ValueError('Event clock currently supports one conversation')
        offset=cache.seen_tokens if incremental else 0
        boundary=None
        if event_clock:
            boundary=(_cognitive_boundary if _cognitive_boundary is not None else
                (cache.cognitive_boundary if incremental and cache.cognitive_boundary is not None else int(decision_positions[0])+offset))
            if type(boundary)is not int or boundary<0:raise ValueError('Explicit public prompt boundary required')
            if incremental:
                if cache.cognitive_boundary is not None and cache.cognitive_boundary!=boundary:raise ValueError('Cached cognitive boundary changed')
                cache.cognitive_boundary=boundary
        if candidate_ids is None:
            if candidate_mask is not None:
                raise ValueError('Candidate mask without descriptions')
            candidates = (None if self.config.hypothesis_mode=='adaptive' else
                          self.core.latent_candidates[None].expand(b,-1,-1))
        else:
            if candidate_mask is None or candidate_ids.shape[0] != b:
                raise ValueError('Candidate descriptions and mask must align')
            candidates = descriptions('candidates',candidate_ids, candidate_mask)
        if goal_ids is None:
            if goal_mask is not None:
                raise ValueError('Goal mask without goal')
            goal = next(self.core.parameters()).new_zeros(b,self.config.width)
            if self.config.cortical_goal_control:
                from .joint_context import selected_goal_context
                goal=selected_goal_context(state)
        else:
            if goal_mask is None or goal_ids.shape[0] != b:
                raise ValueError('Goal descriptions and mask must align')
            goal = descriptions('goal',goal_ids[:,None],goal_mask[:,None]).squeeze(1)
        embeddings = self.body.embed(ids)

        history_ids=(torch.cat((cache.tokens,ids),-1) if incremental and cache.tokens is not None else ids)
        history_mask=torch.ones_like(history_ids) if incremental else mask
        echo_hashes = (causal_ngram_fingerprints(history_ids, history_mask, self.config.echo_ngram)[:,-t:]
                       if self.config.echo_slots > 0 else None)

        def scan(hidden,site):
            encoded = self.to_core(hidden.to(self.to_core.weight.dtype))
            language_prior=None
            choice_boundary=(_cognitive_boundary if _cognitive_boundary is not None else
                (cache.cognitive_boundary if incremental and cache.cognitive_boundary is not None else int(decision_positions[0])+offset))
            if (self.config.language_prior_gain>0 and body_choice_labels and candidate_ids is not None and
                self.config.coupling_mode=='layer_interleaved' and self.config.layer_sites and
                getattr(self.body,'supports_choice_prior',False) and site==self.config.layer_sites[-1] and
                site==len(self.body.decoder.layers)-1):
                if b!=1:raise ValueError('Public choice-prior binding currently requires a single conversation')
                if incremental and cache.choice_prior is not None:
                    fixed_prior=cache.choice_prior
                else:
                    local=choice_boundary-offset
                    if local<0:raise ValueError('Public choice boundary was skipped')
                    fixed_prior=None if local>=t else self.body.choice_label_logits(hidden[:,local:local+1],candidate_ids.shape[1])[:,0]
                    if incremental and fixed_prior is not None:cache.choice_prior=fixed_prior.detach().clone()
                if fixed_prior is not None:language_prior=fixed_prior[:,None,:].expand(-1,t,-1)
            drift = None
            if self.config.body_drift_coupling:
                drift = (drift_level if drift_level is not None
                         else encoded.new_zeros(encoded.shape[0]))
            # Restart at every depth. A previous depth's FINAL sequence state
            # would carry future teacher tokens backward to the first token.
            working = state.working
            regional = (state.cortex,state.relay,state.selection)
            hippocampal=state.hippocampal_population
            cerebellar_prev=state.cerebellar_prediction
            da_prev=state.dopamine
            population=(HypothesisPopulation(encoded.new_zeros(b,self.config.latent_candidates,self.config.width),
                encoded.new_zeros(b,self.config.latent_candidates),encoded.new_zeros(b,self.config.latent_candidates),
                encoded.new_zeros(b,self.config.latent_candidates,self.config.episode_slots))
                if candidates is None else None)
            if candidates is None and _hypothesis_state is not None:
                population=_hypothesis_state
            outputs, shares, routes, reads = [], [], [], []
            start=0
            previous=cache.scans.get(site) if cache is not None else None
            if incremental and previous is not None:
                working,regional,hippocampal,cerebellar_prev,da_prev,population=previous['scratch']
                cache.reused_positions+=cache.seen_tokens
            elif previous is not None:
                length=previous['encoded'].shape[1]
                if (length<t and torch.equal(previous['mask'],mask[:,:length]) and
                    torch.equal(previous['ids'],ids[:,:length]) and torch.equal(previous['encoded'],encoded[:,:length]) and
                    ((language_prior is None and previous.get('language_prior')is None)or(language_prior is not None and previous.get('language_prior')is not None and torch.equal(previous['language_prior'],language_prior[:,:length])))):
                    start=length
                    working,regional,hippocampal,cerebellar_prev,da_prev,population=previous['scratch']
                    outputs,shares,routes,reads=(list(x) for x in previous['surfaces'])
                    cache.reused_positions+=start
                else:cache.prefix_misses+=1
            for position in range(start,t):
                # Language processes all tokens. Full regional dynamics run at
                # public message/query endpoints and subsequent answer tokens.
                # Endpoint is fixed before teacher/generation; no future content
                # controls earlier feedback. Initial token supplies causal shapes.
                before_query=event_clock and offset+position>0 and offset+position<boundary and bool(outputs or previous is not None)
                slow_response=event_clock and offset+position>boundary and (offset+position-boundary)%self.config.response_cognitive_stride!=0
                if before_query or slow_response:
                    step=outputs[-1]if outputs else previous['surfaces'][0][-1]
                    share=torch.zeros_like(step.shared)if before_query else torch.where(mask[:,position,None].bool(),step.shared,torch.zeros_like(step.shared))
                    outputs.append(step);shares.append(share)
                    routes.append(step.route_weights);reads.append(step.candidate_reads)
                    continue
                step = self.core.step(encoded[:,position], working, state, candidates, goal,
                                      read_mode=read_mode, feedback_mode=feedback_mode,
                                      regional_state=regional,circuit_mode=circuit_mode,hypothesis_state=population,
                                      hippocampal_state=hippocampal,hippocampal_mode=hippocampal_mode,
                                      cerebellar_prediction_prev=cerebellar_prev,
                                      thalamic_mode=thalamic_mode,cerebellar_mode=cerebellar_mode,
                                      amygdala_mode=amygdala_mode,da_prev=da_prev,
                                      drift=drift,echo_query_hash=(None if echo_hashes is None else echo_hashes[:,position]),
                                      body_choice_prior=None if language_prior is None or offset+position<choice_boundary else language_prior[:,position],
                                      body_bridge_context=(encoded[:,position] if candidate_ids is not None else None))
                active = mask[:,position,None].bool()
                if step.hypotheses is not None:
                    population=HypothesisPopulation(*(
                        torch.where(active if current.ndim==2 else active[:,:,None],current,getattr(population,key))
                        for key,current in vars(step.hypotheses).items()))
                working = torch.where(active[:,:,None],step.working,working)
                if self.config.regional_circuit:
                    regional = (torch.where(active[:,:,None,None],step.cortex,regional[0]),
                                torch.where(active[:,:,None],step.relay,regional[1]),
                                torch.where(active[:,:,None],step.selection,regional[2]))
                if self.config.hippocampal_circuit:
                    hippocampal=torch.where(active[:,:,None],step.hippocampal_population,hippocampal)
                if self.config.cerebellum:
                    cerebellar_prev=torch.where(active,step.cerebellar_prediction,cerebellar_prev)
                if self.config.dopamine_plasticity:
                    da_prev=torch.where(active[:,0],step.dopamine,da_prev)
                outputs.append(step)
                shares.append(torch.where(active,step.shared,torch.zeros_like(step.shared)))
                routes.append(step.route_weights)
                reads.append(step.candidate_reads)
            if cache is not None:
                cache.executed_positions+=sum(not event_clock or (p==0 and previous is None) or offset+p==0 or (offset+p>=boundary and (offset+p-boundary)%self.config.response_cognitive_stride==0)for p in range(start,t))
                cache.scans[site]=dict(encoded=encoded.detach().clone(),mask=mask.detach().clone(),ids=ids.detach().clone(),
                    language_prior=None if language_prior is None else language_prior.detach().clone(),
                    scratch=(working,regional,hippocampal,cerebellar_prev,da_prev,population),
                    surfaces=(outputs,shares,routes,reads))
            shared = torch.stack(shares,1)
            return outputs, shared, routes, reads, working,regional,hippocampal

        if self.config.coupling_mode == 'layer_interleaved':
            if not hasattr(self.body, 'hidden_interleaved'):
                raise ValueError('Body has no reviewed layer-interleaved adapter')
            latest = []

            def visit(hidden, index):
                result = scan(hidden,index)
                latest[:] = result
                if body_feedback:
                    return hidden + (self.feedback_gain.tanh()*self.to_body(result[1])).to(hidden.dtype)
                return hidden

            if incremental:
                try:hidden=self.body.hidden_interleaved_cached(embeddings,history_mask,visit,self.config.layer_sites,cache)
                except Exception:
                    cache.invalid=True;raise
                cache.tokens=history_ids.detach().clone();cache.seen_tokens+=t
            else:hidden = self.body.hidden_interleaved(embeddings, mask, visit, self.config.layer_sites)
            outputs, shared, routes, reads, working,regional,hippocampal = latest
        else:
            feedback = torch.zeros_like(embeddings)
            for body_round in range(self.config.body_rounds):
                hidden = self.body.hidden(embeddings+feedback, mask)
                outputs, shared, routes, reads, working,regional,hippocampal = scan(hidden,('round',body_round))
                feedback = ((self.feedback_gain.tanh()*self.to_body(shared)).to(embeddings.dtype) if body_feedback
                            else torch.zeros_like(embeddings))
        native = torch.stack([o.native_logits for o in outputs],1)[rows,decision_positions]
        control = torch.stack([o.control_logits for o in outputs],1)[rows,decision_positions]
        outcome = torch.stack([o.predicted_outcomes for o in outputs],1)[rows,decision_positions]
        predicates = (torch.stack([o.predicate_logits for o in outputs],1)[rows,decision_positions]
                      if outputs[0].predicate_logits is not None else None)
        values = torch.stack([o.action_values for o in outputs],1)[rows,decision_positions]
        allocation = torch.stack([o.read_allocation for o in outputs],1)
        hypotheses=(None if outputs[0].hypotheses is None else tuple(
            torch.stack([getattr(o.hypotheses,key) for o in outputs],1)[rows,decision_positions]
            for key in ('vectors','gates','episode_attention')))
        def stack(surface,flag,reduce_last=False):
            if not flag:
                return None
            stacked=torch.stack([getattr(o,surface) for o in outputs],1)
            return stacked[rows,decision_positions] if reduce_last else stacked
        language=self.body.logits(hidden[:,-1:]if _last_token_logits else hidden)
        if self.config.shared_choice_readout and body_choice_labels and candidate_ids is not None:
            # Public option labels are a typed output alphabet. Both outputs
            # project the SAME final, core-modulated cortical hidden state.
            # No generated token/string or answer key enters native decision.
            if not getattr(self.body,'supports_choice_prior',False):raise ValueError('Reviewed public label binding required')
            count=candidate_ids.shape[1]
            labels=self.body.choice_label_tokens[:count]
            if len(labels)!=count:raise ValueError('Public label alphabet exhausted')
            if _last_token_logits:
                if not bool((decision_positions==t-1).all()):raise ValueError('Last-token readout requires last public decision position')
                native=language[:,0].index_select(-1,labels).float()
            else:native=language[rows,decision_positions].index_select(-1,labels).float()
        return StructuralOutput(language,native,control,outcome,shared,
            torch.stack(routes,1),torch.stack(reads,1),working,predicates,values,allocation,
            regional if self.config.regional_circuit else None,
            torch.stack([o.regional_edges for o in outputs],1) if self.config.regional_circuit else None,
            *(hypotheses if hypotheses is not None else (None,None,None)),
            hippocampal,torch.stack([o.hippocampal_mismatch for o in outputs],1)[rows,decision_positions]
            if self.config.hippocampal_circuit else None,
            stack('thalamic',self.config.thalamic_gate),
            stack('cerebellar_prediction',self.config.cerebellum),
            stack('cerebellar_error',self.config.cerebellum),
            stack('dopamine',self.config.dopamine_plasticity),
            stack('neuromod',self.config.neuromodulation),
            stack('drive_vector',self.config.drives),
            stack('initiate_logit',self.config.drives),
            stack('cognitive_control_logits',self.config.cognitive_workspace),body_hidden=hidden,
            soft_candidate_reads=(torch.stack([o.soft_candidate_reads for o in outputs],1)
                if outputs[0].soft_candidate_reads is not None else None),
            hypothesis_selection_logits=(torch.stack([o.hypotheses.selection_logits for o in outputs],1)[rows,decision_positions]
                if outputs[0].hypotheses is not None else None))

    def remember_mental(self,ids,mask,source_ids,source_mask,state,*,kind,status,revision_id=None,
                        goal_metadata='',evidence_count=0,action_goal_id=None,action_known_at=''):
        """Write an explicitly NON-FACT record, preserving every factual slot."""
        self._check(state,ids.shape[0]);self._tokens(ids,mask);self._tokens(source_ids,source_mask)
        if not self.config.cognitive_workspace:raise ValueError('Persistent non-fact workspace is disabled')
        if kind not in KINDS or status not in STATUSES:raise ValueError('Explicit mental origin/status required')
        if revision_id is not None and (type(revision_id) is not int or revision_id<1 or bool((revision_id>state.workspace_seen).any())):
            raise ValueError('Mental revision must refer to an existing past identity')
        if (state.workspace_seen==torch.iinfo(torch.long).max).any():raise OverflowError('Mental journal counter exhausted')
        value=self.encode_descriptions(ids[:,None],mask[:,None]).squeeze(1)
        source=self.encode_descriptions(source_ids[:,None],source_mask[:,None]).squeeze(1)
        kinds=state.workspace_kinds;statuses=state.workspace_statuses
        if revision_id is not None:
            statuses=torch.where(state.workspace_ids==revision_id,
                STATUSES.index('retracted' if status=='retracted' else 'superseded'),statuses)
        goal_statuses=state.goal_statuses
        parent_goal=(state.goal_ids==revision_id if revision_id is not None else torch.zeros_like(state.goal_ids,dtype=torch.bool))
        if bool(parent_goal.any()) and kind not in ('task','goal'):raise ValueError('A non-goal record cannot withdraw an accepted goal')
        if revision_id is not None:
            goal_statuses=torch.where(parent_goal,STATUSES.index('retracted' if status=='retracted' else 'superseded'),goal_statuses)
        goal_changes={'goal_statuses':goal_statuses}
        if kind in ('task','goal') and (status=='active' or bool(parent_goal.any())):
            free=(state.goal_ids<0)|(goal_statuses!=STATUSES.index('active'))
            if bool((~free.any(-1)&~parent_goal.any(-1)).any()):
                raise ValueError('Active goal capacity exhausted; accepted goals cannot be silently evicted')
            slot=torch.where(parent_goal.any(-1),parent_goal.long().argmax(-1),free.long().argmax(-1))
            mask=torch.nn.functional.one_hot(slot,self.config.workspace_slots).bool()
            goal_changes.update(goal_vectors=torch.where(mask[:,:,None],value[:,None],state.goal_vectors),
                goal_sources=torch.where(mask[:,:,None],source[:,None],state.goal_sources),
                goal_ids=torch.where(mask,(state.workspace_seen+1)[:,None],state.goal_ids),
                goal_statuses=torch.where(mask,STATUSES.index(status),goal_statuses))
        staged=replace(state,workspace_statuses=statuses,**goal_changes)
        if goal_metadata:
            if not self.config.goal_joint_control:raise ValueError('Typed goal control is disabled')
            from .goal_metadata import parse_goal_metadata
            data=parse_goal_metadata(goal_metadata,range(evidence_count))
            if evidence_count>self.config.goal_evidence_slots:raise ValueError('Explicit goal evidence capacity exceeded')
            selected_goal=staged.goal_ids==(state.workspace_seen+1)[:,None]
            if not bool(selected_goal.any()):raise ValueError('Typed metadata requires an actual protected goal write')
            updates={}
            values=dict(goal_priority=data['priority'],goal_icost=data['interruption_cost'],
                goal_deadline_at=data['deadline_at']or 0.,goal_deadline_mask=int(data['deadline_at']is not None),
                goal_accept=('pending','accepted','declined').index(data['acceptance']),
                goal_sources_typed=('user_message','device_log','document','tool_result','model_output','schedule').index(data['source_kind']),
                goal_ev_ok=data['evidence_ok'],goal_ev_open=data['evidence_open'],goal_can_act=int(data['can_act']))
            for name,v in values.items():
                old=getattr(staged,name);updates[name]=torch.where(selected_goal,torch.full_like(old,v),old)
            refs=torch.arange(self.config.goal_evidence_slots,device=ids.device)<evidence_count
            updates['goal_evidence']=torch.where(selected_goal[:,:,None],refs[None,None],staged.goal_evidence)
            revision=staged.goal_revision+1 if revision_id is not None else torch.zeros_like(staged.goal_revision)
            updates['goal_revision']=torch.where(selected_goal,revision,staged.goal_revision)
            staged=replace(staged,**updates)
        if action_goal_id is not None:
            if not self.config.goal_joint_control:raise ValueError('Typed goal action receipts are disabled')
            selected_goal=(staged.goal_ids==action_goal_id)&(staged.goal_statuses==STATUSES.index('active'))
            if not bool(selected_goal.any()):raise ValueError('Execution receipt must identify a current active goal')
            now=datetime.fromisoformat(action_known_at).timestamp()
            staged=replace(staged,goal_attempts=staged.goal_attempts+selected_goal.long(),
                goal_last_action=torch.where(selected_goal,torch.full_like(staged.goal_last_action,now),staged.goal_last_action))
        old=self.core.workspace.decorated(staged)
        proposal=value+self.core.workspace.kind.weight[KINDS.index(kind)]+self.core.workspace.status.weight[STATUSES.index(status)]
        decision=self.core.episodic_policy(torch.nn.functional.normalize(value,dim=-1),proposal,source,
            state.working[:,-1]+self.core.workspace.goal_context(staged),
            torch.nn.functional.normalize(state.workspace_vectors,dim=-1),old,state.workspace_sources,
            state.workspace_ids>0,state.workspace_ids,state.workspace_seen,stage=2)
        selected=decision.write_weights[:,:,None]
        hard=selected.detach().squeeze(-1).bool()
        new=replace(staged,workspace_vectors=state.workspace_vectors*(1-selected)+value[:,None]*selected,
            workspace_sources=state.workspace_sources*(1-selected)+source[:,None]*selected,
            workspace_ids=torch.where(hard,(state.workspace_seen+1)[:,None],state.workspace_ids),
            workspace_kinds=torch.where(hard,KINDS.index(kind),kinds),
            workspace_statuses=torch.where(hard,STATUSES.index(status),statuses),
            workspace_seen=state.workspace_seen+1,
            workspace_admissions=state.workspace_admissions+(decision.action>0).long())
        from .mental_archive import commit as archive_mental
        new=archive_mental(self,state,new,value,source,kind,status,revision_id)
        self._check(new,ids.shape[0]);return new

    def goal_action_inputs(self,state,tick):
        """(features (1,R,29), context (1,9), legality (1,R,A)) for one joint head.

        Typed inputs and arithmetic only. The mask carries slot occupancy,
        autonomy and the remaining resource; it cannot read a deadline, an
        urgency, a priority or an attempt count, so no hand-written
        "time passed -> speak" rule can live here.
        """
        if not self.config.goal_joint_control:raise ValueError('Goal-joint control is disabled')
        slots=self.config.workspace_slots
        now=float(datetime.fromisoformat(tick.known_at).timestamp())
        features=goal_feature_rows(state.goal_ids[0],state.goal_statuses[0],state.goal_accept[0],
            state.goal_sources_typed[0],state.goal_priority[0],state.goal_icost[0],
            state.goal_deadline_at[0],state.goal_deadline_mask[0],state.goal_attempts[0],
            state.goal_last_action[0],state.goal_evidence[0],state.goal_ev_ok[0],state.goal_ev_open[0],
            state.goal_revision[0],state.goal_can_act[0],now,slots=slots)
        context=context_features(state.goal_ids[0],state.goal_statuses[0],state.goal_accept[0],
            state.goal_priority[0],state.goal_deadline_at[0],state.goal_deadline_mask[0],now,
            tick.elapsed_seconds,float(state.elapsed_seconds[0]),
            tick.interruption_cost,tick.resource,slots=slots)
        active=(state.goal_ids>0)&(state.goal_statuses==STATUSES.index('active'))
        occupied=torch.cat((active,torch.ones_like(state.goal_ids[:,:1],dtype=torch.bool)),1)
        permitted=torch.cat(((state.goal_can_act>0),torch.zeros_like(state.goal_can_act[:,:1],dtype=torch.bool)),1)
        allowed=legality_mask(occupied,can_act=permitted,resource=float(tick.resource),
                              act_resource_floor=self.config.goal_act_resource_floor)
        accepted=torch.cat((state.goal_accept==1,torch.ones_like(state.goal_accept[:,:1],dtype=torch.bool)),1)
        for action in('speak','act'):allowed[:,:,ACTIONS.index(action)]&=accepted
        if self.config.contextual_goal_control:
            from .joint_context import resource_legal
            allowed=resource_legal(allowed,float(tick.resource))
        return features.unsqueeze(0),context,allowed

    def cognitive_tick(self,state,tick,*,cortical_read=None,replay_action=None,replay_goal=None,drift_level=None,**forward_options):
        """Time creates an opportunity; a neural policy selects its operation.

        Think may update working/relay/selection/read populations. It never
        commits a factual event or changes observation plasticity/TD targets.
        A straight-through think gate is a training surrogate, not an oracle.
        """
        self._check(state,1)
        if not self.config.cognitive_workspace or not isinstance(tick,CognitiveTick):raise ValueError('Explicit cognitive opportunity required')
        cortical_active=self.config.cortical_goal_control and replay_action is None and tick.resource>0
        if cortical_active:
            from .cortical_control import CorticalOpportunityRead
            if not isinstance(cortical_read,CorticalOpportunityRead):raise ValueError('Same-model native opportunity read required; use native.tick')
            cortical_read.validate(self,state,tick)
        if (state.cognitive_ticks==torch.iinfo(torch.long).max).any():raise OverflowError('Cognitive opportunity counter exhausted')
        elapsed=state.elapsed_seconds+tick.elapsed_seconds
        if not bool(torch.isfinite(elapsed).all()):raise ValueError('Elapsed time overflow')
        from .cortical_control import effective_opportunity
        effective=effective_opportunity(state,tick)
        remaining=effective.deadline_remaining
        context=effective.features(state.working,float(elapsed[0]))
        drives=state.drive_state
        if drives is not None:
            tau=self.core.workspace.idle_tau_log.clamp(0.,18.).exp()
            idle=(drives[:,2]+tick.elapsed_seconds/tau).clamp(max=1.)
            drives=torch.stack((drives[:,0],drives[:,1],idle),-1)
        staged=replace(state,elapsed_seconds=elapsed,deadline_seconds=state.working.new_tensor([remaining or 0.]),cognitive_context=context,drive_state=drives)
        if replay_goal is not None and (type(replay_goal) is not int or not 0<=replay_goal<=self.config.workspace_slots):
            raise ValueError('Historical goal row must be an actual recorded row')
        if self.config.goal_joint_control:
            features,context,allowed=self.goal_action_inputs(staged,tick)
            if cortical_active:
                from .joint_context import joint_context_logits
                from .cortical_control import cortical_joint_policy
                residual,_=joint_context_logits(self.core.workspace,staged,features,context)
                calibration=None
                if self.config.cortical_control_calibration:
                    calibration=self.core.workspace.cortical_calibration(cortical_read.logits,staged,features,context,allowed)
                decision=cortical_joint_policy(cortical_read.logits,residual,allowed,null_row=self.config.workspace_slots,calibration=calibration)
            else:decision=self.core.workspace.goal_action(staged,features,context,allowed)
            joint=decision['probabilities'];p=joint.sum(-2)
            if replay_goal is not None:
                row=torch.full((1,),replay_goal,device=joint.device,dtype=torch.long)
                act=torch.full((1,),0 if replay_action is None else replay_action,
                               device=joint.device,dtype=torch.long)
            elif replay_action is not None:
                act=torch.full((1,),replay_action,device=joint.device,dtype=torch.long)
                row=joint[:,:,replay_action].argmax(-1)
            else:
                row=decision['row'];act=decision['action']
            changes={'cognitive_joint':joint.detach(),'cognitive_goal':row.detach()}
            if self.config.cortical_goal_control:
                identity=staged.goal_ids.gather(1,row.clamp_max(self.config.workspace_slots-1)[:,None]).squeeze(1)
                changes['cognitive_goal_identity']=torch.where(row<self.config.workspace_slots,identity,torch.full_like(identity,-1)).detach()
        else:
            logits=self.core.workspace.control_logits(staged);p=logits.softmax(-1)
            action=(logits.argmax(-1) if replay_action is None else torch.full((1,),replay_action,device=logits.device,dtype=torch.long))
            act=action
            changes={}
        hard=(act==ACTIONS.index('think')).to(p.dtype)
        gate=(hard+(p[:,1]-p[:,1].detach()) if torch.is_grad_enabled() and replay_action is None else hard)
        if torch.is_grad_enabled() or bool(hard.any()):
            allowed={k:v for k,v in forward_options.items() if k in ('read_mode','feedback_mode','circuit_mode',
                'hippocampal_mode','thalamic_mode','cerebellar_mode','amygdala_mode')}
            goal=self.core.workspace.goal_context(staged)
            if self.config.contextual_goal_control:
                from .joint_context import focused_goal
                goal=focused_goal(staged,row)
            step=self.core.step(state.working[:,-1],state.working,staged,None,
                goal,resource=state.working.new_tensor([[tick.resource]]),
                drift=(state.working.new_zeros(1) if self.config.body_drift_coupling and drift_level is None else drift_level),**allowed)
            for name,value in (('working',step.working),('cortex',step.cortex),('relay',step.relay),
                ('selection',step.selection),('hippocampal_population',step.hippocampal_population),('thalamic',step.thalamic)):
                old=getattr(state,name)
                if value is not None and old is not None:
                    g=gate.reshape(-1,*([1]*(old.ndim-1)))
                    changes[name]=old*(1-g)+value*g
        if drives is not None:
            changes['drive_state']=torch.stack((drives[:,0],drives[:,1],drives[:,2]*(1-gate)),-1)
        new=replace(staged,**changes,cognitive_ticks=state.cognitive_ticks+1,
            deliberations=state.deliberations+hard.long(),cognitive_distribution=p)
        self._check(new,1);return new

    def ingest(self, observation_ids, observation_mask, source_ids, source_mask,
               state, *, origin='observation', **forward_options):
        if origin != 'observation':
            raise ValueError('Hypotheses/teacher answers cannot commit factual experience')
        self._tokens(observation_ids, observation_mask)
        self._tokens(source_ids, source_mask)
        if observation_ids.shape[0] != source_ids.shape[0]:
            raise ValueError('Source/observation batch mismatch')
        output = self(observation_ids, observation_mask, state,_last_token_logits=True, **forward_options)
        return self.commit_observation_output(observation_ids,observation_mask,source_ids,source_mask,state,output,**forward_options)

    def commit_observation_output(self,observation_ids,observation_mask,source_ids,source_mask,state,output,
                                  *,echo_ids=None,**forward_options):
        """Commit an actual observed prefill's populations, without a second body pass.

        The native session supplies its own completed prefill. This changes the
        observation encoder ABI intentionally: native body history participates
        in perception. Old nonempty state is never adopted under the new ABI.
        """
        self._check(state,observation_ids.shape[0]);self._tokens(observation_ids,observation_mask)
        self._tokens(source_ids,source_mask)
        if output.shared_hidden.shape[:2]!=observation_ids.shape or output.body_hidden.shape[:2]!=observation_ids.shape:
            raise ValueError('Actual observed prefill output/positions required')
        weights = observation_mask.to(output.shared_hidden.dtype)[:,:,None]
        encoded = (output.shared_hidden*weights).sum(1)/weights.sum(1)
        if self.config.cognitive_clock=='event':
            last=torch.arange(observation_ids.shape[1],device=observation_ids.device)[None].expand_as(observation_ids)
            last=last.masked_fill(~observation_mask.bool(),-1).amax(-1)
            encoded=output.shared_hidden[torch.arange(observation_ids.shape[0],device=observation_ids.device),last]
        if self.config.perception_memory:
            last=torch.arange(observation_ids.shape[1],device=observation_ids.device)[None].expand_as(observation_ids)
            last=last.masked_fill(~observation_mask.bool(),-1).amax(-1)
            perceived=output.body_hidden[torch.arange(observation_ids.shape[0],device=observation_ids.device),last]
            encoded=self.to_core(perceived.to(self.to_core.weight.dtype))
        source = self.encode_descriptions(source_ids[:,None],source_mask[:,None]).squeeze(1)
        options=dict(regions=output.scratch_regions,hippocampal_population=output.scratch_hippocampal)
        if 'hippocampal_mode' in forward_options:
            options['hippocampal_mode']=forward_options['hippocampal_mode']
        if self.config.echo_slots > 0:
            echo_tokens=observation_ids if echo_ids is None else echo_ids
            fingerprints = causal_ngram_fingerprints(echo_tokens, torch.ones_like(echo_tokens)if echo_ids is not None else observation_mask,
                                                       self.config.echo_ngram)
            options['echo_hash_in'] = fingerprints[:, -1]
        if self.config.r8_active:
            rows=torch.arange(observation_ids.shape[0],device=encoded.device)
            positions=torch.arange(observation_mask.shape[1],device=encoded.device)[None]
            last=positions.masked_fill(~observation_mask.bool(),-1).amax(-1)
            def last_surface(surface):
                stacked=getattr(output,surface)
                return None if stacked is None else stacked[rows,last]
            mismatch=(last_surface('cerebellar_errors') if self.config.cerebellum else
                      (output.hippocampal_mismatch if self.config.hippocampal_circuit
                       else None))
            allocation=last_surface('read_allocation')
            options.update(thalamic=last_surface('thalamic'),
                cerebellar_prediction=last_surface('cerebellar_prediction'),
                dopamine=last_surface('dopamine'),neuromod=last_surface('neuromod'),
                mismatch=mismatch,
                uncertainty=(None if allocation is None else
                             -(allocation*allocation.clamp_min(1e-12).log()).sum(-1)))
        return self.core.commit_observation(encoded,source,output.scratch_working,state,**options)

    def consolidate(self, state,*,hippocampal_mode='normal'):
        self._check(state, state.working.shape[0])
        return self.core.consolidate(state,hippocampal_mode=hippocampal_mode)

    @torch.no_grad()
    def drive_report(self, state) -> dict:
        """Model-internal causation surface for spontaneous behavior.

        The DECISION to speak comes from the model's own drive state and the
        initiate head evaluated here; the runtime only executes what this
        returns (brainstem analogue — execution, not causation). No external
        scheduler decides when Treyrin talks.
        """
        self._check(state, state.working.shape[0])
        if self.config.cognitive_workspace:
            if self.config.goal_joint_control:
                table=state.cognitive_joint[0];choice=int(table.flatten().argmax());row=choice//len(ACTIONS);action=choice%len(ACTIONS)
                return dict(action=ACTIONS[action],goal_slot=row,
                    goal_id=None if row==self.config.workspace_slots else int(state.goal_ids[0,row]),
                    probabilities=table.sum(0).cpu().tolist(),joint_probabilities=table.cpu().tolist(),self_initiates=action>=2,
                    cognitive_ticks=int(state.cognitive_ticks[0]),deliberations=int(state.deliberations[0]),
                    elapsed_seconds=float(state.elapsed_seconds[0]),calibrated=False,scope='last actual cognitive opportunity')
            p=self.core.workspace.control_logits(state).softmax(-1)[0]
            action=int(p.argmax())
            return dict(action=ACTIONS[action],probabilities=p.cpu().tolist(),self_initiates=action>=2,
                cognitive_ticks=int(state.cognitive_ticks[0]),deliberations=int(state.deliberations[0]),
                elapsed_seconds=float(state.elapsed_seconds[0]),calibrated=False)
        if not self.config.drives:
            raise ValueError('Self-initiation requires the hypothalamic drives')
        d = state.drive_state[0]
        curiosity, error_pressure, idle = float(d[0]), float(d[1]), float(d[2])
        vector, logit = self.core.drives(d[None])
        return {"curiosity": curiosity, "error_pressure": error_pressure,
                "idle": idle, "initiate_logit": float(logit[0]),
                "self_initiates": bool(logit[0] > 0.0)}

    @torch.no_grad()
    def generate(self, prompt_ids, state, *, max_new_tokens=None, eos_token_id=None,
                 candidate_ids=None, candidate_mask=None, goal_ids=None, goal_mask=None,
                 sample=False, temperature=1.0, generator=None,use_prefix_cache=True,use_decoder_cache=False,cache_diagnostics=None, **forward_options):
        if self.training:
            raise ValueError('Set eval mode for reproducible generation')
        if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens < 1):
            raise ValueError('Generation budget must be None or a positive int')
        if temperature <= 0:
            raise ValueError('Positive sampling temperature required')
        from generation_result import position_capacity,GenerationResult
        capacity=position_capacity(self,None,prompt_ids)
        if max_new_tokens is None and (capacity is None or capacity<1):
            raise ValueError('Unbounded generation requires a real model position ceiling')
        ids = prompt_ids
        cache=AutoregressiveDecodeCache() if use_decoder_cache else (InferencePrefixCache() if use_prefix_cache else None)
        if eos_token_id is None:
            base=getattr(getattr(self,'body',None),'model',None)
            registered=getattr(getattr(base,'generation_config',None),'eos_token_id',None)
            if registered is None:registered=getattr(getattr(base,'config',None),'eos_token_id',None)
            eos_token_id=registered
        index=0;finish='length';eos=False
        eos_ids=set(eos_token_id if isinstance(eos_token_id,(list,tuple,set)) else (() if eos_token_id is None else (eos_token_id,)))
        base=getattr(getattr(self,'body',None),'model',None)
        for config in (getattr(base,'generation_config',None),getattr(base,'config',None)):
            registered=getattr(config,'eos_token_id',None)
            if registered is not None:eos_ids.update(registered if isinstance(registered,(list,tuple,set)) else (registered,))
        while True:
            if max_new_tokens is not None and index>=max_new_tokens:
                break
            if capacity is not None and index>=capacity:
                finish='context_limit';break
            current=ids[:,-1:] if use_decoder_cache and index else ids
            scratch={'_decode_cache':cache} if use_decoder_cache else {'_inference_cache':cache}
            output = self(current,torch.ones_like(current),state,candidate_ids=candidate_ids,
                          candidate_mask=candidate_mask,goal_ids=goal_ids,goal_mask=goal_mask,
                          _cognitive_boundary=prompt_ids.shape[1]-1,**scratch,**forward_options)
            logits = output.language_logits[:,-1]
            token = (torch.multinomial((logits/temperature).softmax(-1),1,generator=generator)
                     if sample else logits.argmax(-1,keepdim=True))
            ids = torch.cat((ids,token),-1); index+=1
            if token.item() in eos_ids:
                eos=True;finish='stop';break
        self.last_generation=GenerationResult(ids[0,prompt_ids.shape[1]:].tolist(),finish,eos,max_new_tokens,cache,capacity)
        if cache_diagnostics is not None:
            cache_diagnostics.update(reused_positions=cache.reused_positions if cache else 0,
                executed_positions=cache.executed_positions if cache else 0,prefix_misses=cache.prefix_misses if cache else 0,
                finish_reason=finish,eos_reached=eos,capacity=capacity,generation_budget=max_new_tokens)
        return ids

    @torch.no_grad()
    def propose_and_judge(self, prompt_ids, state, *, max_hypothesis_tokens=None,
                          count=4, goal_ids=None, goal_mask=None, generator=None,use_decoder_cache=False,**forward_options):
        """Initial no-supplied-candidate path; self-generated alternatives + controls.

        This does not establish useful intuition/diversity. Generated tokens stay
        in scratch; callers must not commit them as observations.
        """
        if type(count) is not int or count < 2:
            raise ValueError('At least two self-generated hypotheses required')
        hypotheses = []
        for index in range(count):
            generated = self.generate(prompt_ids,state,max_new_tokens=max_hypothesis_tokens,
                goal_ids=goal_ids,goal_mask=goal_mask,sample=index>0,generator=generator,
                use_decoder_cache=use_decoder_cache,**forward_options)
            hypotheses.append(generated[:,prompt_ids.shape[1]:])
        self.last_hypothesis_lengths=[int(h.shape[1]) for h in hypotheses]
        hypothesis = torch.nn.utils.rnn.pad_sequence([h[0] for h in hypotheses],batch_first=True,padding_value=0)[None]
        hypothesis_mask=(torch.arange(hypothesis.shape[-1],device=hypothesis.device)[None,None]<torch.tensor(self.last_hypothesis_lengths,device=hypothesis.device)[None,:,None]).long()
        # Sampling is not evidence of novelty/usefulness; duplicates remain visible.
        output = self(prompt_ids,torch.ones_like(prompt_ids),state,
            candidate_ids=hypothesis,candidate_mask=hypothesis_mask,
            goal_ids=goal_ids,goal_mask=goal_mask,**forward_options)
        return hypothesis, output

    def export_bundle(self, directory, state):
        self._check(state,state.working.shape[0])
        path = Path(directory)
        path.mkdir(parents=True,exist_ok=False)
        torch.save(self.state_dict(),path/'weights.pt')
        torch.save({**{k:v.detach().cpu() for k,v in state.tensors().items()},
                    'binding':state.binding},path/'experience.pt')
        manifest = dict(schema='treyrin-structural-0.16',config=asdict(self.config),
            body_asset=self.body.asset_identity,binding=self._binding,
            capability_accepted=False,files={})
        for name in ('weights.pt','experience.pt'):
            manifest['files'][name] = hashlib.sha256((path/name).read_bytes()).hexdigest()
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        return manifest

    def load_bundle(self, directory):
        """Caller constructs the reviewed body class; no arbitrary class loading."""
        path = Path(directory)
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('schema') != 'treyrin-structural-0.16' or
            manifest.get('config') != json.loads(json.dumps(asdict(self.config))) or
            manifest.get('body_asset') != self.body.asset_identity):
            raise ValueError('Bundle architecture/body differs')
        expected = {'weights.pt','experience.pt'}
        if set(manifest.get('files',{})) != expected:
            raise ValueError('Unexpected bundle asset list')
        for name in expected:
            if hashlib.sha256((path/name).read_bytes()).hexdigest() != manifest['files'][name]:
                raise ValueError('Bundle asset hash differs')
        self.load_state_dict(torch.load(path/'weights.pt',map_location='cpu',weights_only=True))
        if self.seal_parameters() != manifest['binding']:
            raise ValueError('Parameter content binding differs')
        data = torch.load(path/'experience.pt',map_location='cpu',weights_only=True)
        ref = next(self.parameters())
        state = ExperienceState(**{k:(v.to(ref.device) if isinstance(v,Tensor) else v)
                                   for k,v in data.items()})
        self._check(state,state.working.shape[0])
        return state
