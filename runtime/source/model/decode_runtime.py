"""One causal prefill shared by native output, generation and all continuations."""
import copy
import torch
from generation_result import GenerationResult,position_capacity
from treyrin.structural_model import AutoregressiveDecodeCache

class DecodeEpisode:
    def __init__(self,model,state,prompt_ids,*,candidate_ids=None,candidate_mask=None,**options):
        if torch.is_grad_enabled():raise ValueError('DecodeEpisode is inference-only')
        self.model=model;self.state=state;self.ids=prompt_ids
        self.options=dict(candidate_ids=candidate_ids,candidate_mask=candidate_mask,**options)
        self.cache=AutoregressiveDecodeCache()
        self.output=model(prompt_ids,torch.ones_like(prompt_ids),state,_decode_cache=self.cache,**self.options)
        self.first_logits=self.output.language_logits[:,-1].detach().clone()
        self.last_generation=None

    def fork(self):return copy.deepcopy(self.cache)

    def position_capacity(self):return position_capacity(self.model,self.cache,self.ids)

    def generate(self,max_new_tokens=None,*,eos_token_id=None,allowed_first_tokens=None):
        from generation_result import GenerationResultMixin
        self.last_generation=GenerationResultMixin.generate_result(self,max_new_tokens,
            eos_token_id=eos_token_id,allowed_first_tokens=allowed_first_tokens)
        return self.last_generation.token_ids

    def log_probabilities(self,continuation):
        if not continuation:raise ValueError('Nonempty complete continuation required')
        values=[self.first_logits.float().log_softmax(-1)[0,continuation[0]]]
        if len(continuation)>1:
            ids=torch.tensor([continuation[:-1]],device=self.ids.device)
            output=self.model(ids,torch.ones_like(ids),self.state,_decode_cache=self.fork(),**self.options)
            probs=output.language_logits[0].float().log_softmax(-1)
            targets=torch.tensor(continuation[1:],device=self.ids.device)
            values+=list(probs.gather(1,targets[:,None]).squeeze(1).unbind())
        return torch.stack(values)
