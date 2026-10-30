"""Generation transport metadata; ending by a cap is never an EOS success."""
from dataclasses import dataclass
import torch

def position_capacity(model,cache,ids):
    """Remaining positions in the LOADED body; None is never replaced by a number.

    The ceiling is the real config limit minus every position already consumed by
    the prefill. A native capsule can consume most of the window before the first
    answer token, so the prompt length alone overstates the remaining room. None
    means this body exposes no position limit: callers must then supply an explicit
    budget instead of asking for unbounded generation.
    """
    body=getattr(model,'body',None)
    base=getattr(body,'model',None)
    if base is None:base=model
    max_positions=getattr(getattr(base,'config',None),'max_position_embeddings',None)
    if max_positions is None:return None
    if type(max_positions)is not int or max_positions<1:
        raise ValueError('Loaded body position ceiling must be a positive int')
    consumed=max(int(getattr(cache,'seen_tokens',0)),int(ids.shape[1]))
    # Native KV is a sliding window. Its absolute lifetime counter is not
    # resident context and must not expire a permanent conversation at 262k.
    window=getattr(cache,'native_window',None)
    if isinstance(window,int) and window>0:
        consumed=min(consumed,window)
    return max_positions-consumed

@dataclass
class GenerationResult:
    token_ids:list[int]
    finish_reason:str
    eos_reached:bool
    budget:int|None
    cache:object
    capacity:int|None=None

class GenerationInterrupted(KeyboardInterrupt):
    def __init__(self,result):
        self.result=result
        super().__init__('Generation interrupted; committed input and sampled prefix retained')

class GenerationResultMixin:
    def position_capacity(self):
        return position_capacity(self.model,self.cache,self.ids)

    def generate_result(self,max_new_tokens=None,*,eos_token_id=None,allowed_first_tokens=None,
                        reserved_tokens=0,on_token=None,should_cancel=None,prefix_tokens=(),
                        sample=False,temperature=1.,generator=None):
        if max_new_tokens is not None and (type(max_new_tokens)is not int or max_new_tokens<1):
            raise ValueError('Generation budget must be None or a positive int')
        cache=self.fork();logits=self.first_logits;tokens=[];eos=False
        if allowed_first_tokens is not None:
            allowed=list(allowed_first_tokens)
            if not allowed or len(set(allowed))!=len(allowed)or any(type(i)is not int or not 0<=i<logits.shape[-1]for i in allowed):
                raise ValueError('Explicit valid public output-token set required')
            constrained=torch.full_like(logits,float('-inf'));constrained[:,allowed]=logits[:,allowed];logits=constrained
        eos_ids=set(eos_token_id if isinstance(eos_token_id,(list,tuple,set)) else (() if eos_token_id is None else (eos_token_id,)))
        base=getattr(getattr(self.model,'body',None),'model',None)
        for config in (getattr(base,'generation_config',None),getattr(base,'config',None)):
            registered=getattr(config,'eos_token_id',None)
            if registered is not None:eos_ids.update(registered if isinstance(registered,(list,tuple,set)) else (registered,))
        capacity=self.position_capacity()
        if type(reserved_tokens)is not int or reserved_tokens<0:raise ValueError('Nonnegative transport reserve required')
        if capacity is not None:capacity=max(0,capacity-reserved_tokens)
        # Position capacity is diagnostics/resource metadata, never an answer
        # terminator. Only EOS/EOT, explicit cancellation or a real runtime
        # failure ends the response.
        i=0;finish='length'
        prefix=list(prefix_tokens)
        try:
            while True:
                if should_cancel is not None and should_cancel():finish='cancelled';break
                if max_new_tokens is not None and i>=max_new_tokens:finish='length';break
                if temperature<=0:raise ValueError('Positive generation temperature required')
                token=(prefix[i] if i<len(prefix) else int(torch.multinomial((logits.float()/temperature).softmax(-1),1,generator=generator)[0,0]) if sample else int(logits.argmax(-1)[0]))
                if type(token)is not int or not 0<=token<logits.shape[-1]:raise ValueError('Invalid saved generation prefix')
                ids=torch.tensor([[token]],device=self.ids.device)
                output=self.model(ids,torch.ones_like(ids),self.state,_decode_cache=cache,**self.options)
                logits=output.language_logits[:,-1]
                tokens.append(token);eos=token in eos_ids;i+=1
                if on_token is not None and i>len(prefix):on_token(token,list(tokens))
                if eos:finish='stop';break
        except KeyboardInterrupt as exc:
            raise GenerationInterrupted(GenerationResult(tokens,'cancelled',False,max_new_tokens,cache,capacity)) from exc
        return GenerationResult(tokens,finish,eos,max_new_tokens,cache,capacity)

    def generate(self,max_new_tokens=None,*,eos_token_id=None,allowed_first_tokens=None,**kwargs):
        result=self.generate_result(max_new_tokens,eos_token_id=eos_token_id,allowed_first_tokens=allowed_first_tokens,**kwargs)
        self.last_generation=result
        return result.token_ids
