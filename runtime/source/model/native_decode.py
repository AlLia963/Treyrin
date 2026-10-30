"""Cortical request scratch seeded by immutable per-conversation native state."""
import copy
import torch
from decode_runtime import DecodeEpisode
from native_stream import Capsule
from treyrin.structural_model import AutoregressiveDecodeCache

class NativeEpisode(DecodeEpisode):
    def __init__(self,model,state,prompt_ids,capsule,*,candidate_ids=None,candidate_mask=None,prefill_chunk=448,**options):
        if torch.is_grad_enabled():raise ValueError('Native episode is inference-only')
        if capsule.encoder_binding!=model._binding:raise ValueError('Native experience/parameter content differs')
        self.model=model;self.state=state;self.ids=prompt_ids
        self.options=dict(candidate_ids=candidate_ids,candidate_mask=candidate_mask,**options)
        self.cache=AutoregressiveDecodeCache();self.cache.body_cache=copy.deepcopy(capsule.cache)
        self.cache.seen_tokens=capsule.seen;self.cache.native_window=capsule.window
        if type(prefill_chunk)is not int or prefill_chunk<1:raise ValueError('Bounded prefill chunk required')
        self.options['_cognitive_boundary']=capsule.seen+prompt_ids.shape[1]-1
        for start in range(0,prompt_ids.shape[1],prefill_chunk):
            part=prompt_ids[:,start:start+prefill_chunk]
            self.output=model(part,torch.ones_like(part),state,_decode_cache=self.cache,_last_token_logits=True,**self.options)
        self.first_logits=self.output.language_logits[:,-1].detach().clone()

    def commit_observed_prefix(self):
        # Generation forks this scratch. Only the actually observed prefill is
        # committed; hypothetical continuations are never factual experience.
        return Capsule(self.cache.body_cache,self.cache.seen_tokens,self.cache.native_window,self.model._binding)
