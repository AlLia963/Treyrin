"""Per-conversation native Qwen KV/GDN state, bounded KV and absolute positions.

Prototype only: persistent native state is separate from query scratch. No text
retrieval, repeated journal prompt, extra language model or hidden global cache.
This does not itself prove useful retention beyond the attention window.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from transformers.cache_utils import DynamicCache,DynamicLayer,DynamicSlidingWindowLayer,LinearAttentionLayer
from transformers.masking_utils import create_causal_mask,create_recurrent_attention_mask

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def empty_cache(config,window):
    if type(window)is not int or window<2:raise ValueError('A bounded positive native attention window >=2 required')
    cache=DynamicCache(config=config)
    for i,layer in enumerate(cache.layers):
        if type(layer)is DynamicLayer:cache.layers[i]=DynamicSlidingWindowLayer(window)
        elif type(layer)is not LinearAttentionLayer:raise ValueError('Unreviewed native cache layer')
    return cache

@dataclass(frozen=True)
class Capsule:
    cache:object
    seen:int
    window:int
    encoder_binding:str

    @classmethod
    def empty(cls,body,*,window,binding):
        if not isinstance(binding,str)or len(binding)!=64:raise ValueError('Content encoder binding required')
        return cls(empty_cache(body.decoder.config,window),0,window,binding)

    def tensors(self):
        out={}
        for i,layer in enumerate(self.cache.layers):
            if type(layer)is DynamicSlidingWindowLayer and layer.is_initialized:
                out[f'{i}.keys']=layer.keys;out[f'{i}.values']=layer.values
            elif type(layer)is LinearAttentionLayer:
                for n in range(layer.number_of_states):
                    for name in ('conv_states','recurrent_states'):
                        value=getattr(layer,name)[n]
                        if value is not None:out[f'{i}.{name}.{n}']=value
        return out

    def save(self,path):
        root=Path(path);root.mkdir(exist_ok=False)
        save_file({n:t.detach().cpu().contiguous().clone()for n,t in self.tensors().items()},root/'native.safetensors')
        layers=[]
        for layer in self.cache.layers:
            if type(layer)is DynamicSlidingWindowLayer:
                layers.append(dict(kind='kv',initialized=layer.is_initialized,cumulative_length=layer.cumulative_length))
            else:
                layers.append(dict(kind='gdn',states=layer.number_of_states,has_previous_state=layer.has_previous_state,
                    conv_kernel_size=layer.conv_kernel_size))
        manifest=dict(schema='treyrin-native-capsule-v1',seen=self.seen,window=self.window,encoder_binding=self.encoder_binding,
            implementation_sha256=digest(__file__),payload_sha256=digest(root/'native.safetensors'),layers=layers)
        (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2),encoding='utf-8')

    @classmethod
    def restore(cls,path,body,*,binding):
        root=Path(path);m=json.loads((root/'manifest.json').read_text('utf-8'))
        if m['schema']!='treyrin-native-capsule-v1'or m['encoder_binding']!=binding or m['implementation_sha256']!=digest(__file__):
            raise ValueError('Native cache encoder/implementation differs; never retag old experience')
        if m['payload_sha256']!=digest(root/'native.safetensors'):raise ValueError('Native state payload changed')
        cache=empty_cache(body.decoder.config,m['window']);values=load_file(root/'native.safetensors',device=str(next(body.parameters()).device))
        if len(m['layers'])!=len(cache.layers)or type(m['seen'])is not int or m['seen']<0:raise ValueError('Native layer/count schema differs')
        expected=set()
        for i,(layer,meta)in enumerate(zip(cache.layers,m['layers'])):
            if type(layer)is DynamicSlidingWindowLayer:
                if meta['kind']!='kv'or meta['cumulative_length']!=m['seen']:raise ValueError('Native absolute KV positions differ')
                if meta['initialized']:
                    k,v=values[f'{i}.keys'],values[f'{i}.values'];expected.update([f'{i}.keys',f'{i}.values'])
                    if k.shape!=v.shape or k.ndim!=4 or k.shape[0]!=1 or k.shape[-2]!=min(m['seen'],m['window']-1):
                        raise ValueError('Native bounded KV shape differs')
                    layer.lazy_initialization(k,v);layer.keys=k;layer.values=v
                elif m['seen']:raise ValueError('Missing nonempty KV state')
                layer.cumulative_length=m['seen']
            else:
                if meta['kind']!='gdn'or meta['states']!=layer.number_of_states:raise ValueError('Native GDN schema differs')
                for n in range(layer.number_of_states):
                    cs=values.get(f'{i}.conv_states.{n}');rs=values.get(f'{i}.recurrent_states.{n}')
                    if cs is not None:expected.add(f'{i}.conv_states.{n}')
                    if rs is not None:expected.add(f'{i}.recurrent_states.{n}')
                    if m['seen']and(cs is None or rs is None):raise ValueError('Missing GDN state')
                    if cs is not None or rs is not None:
                        layer.lazy_initialization(conv_states=cs,recurrent_states=rs,state_idx=n,
                            conv_kernel_size=meta['conv_kernel_size'][str(n)])
                        if cs is not None:layer.conv_states[n]=cs
                        if rs is not None:layer.recurrent_states[n]=rs
                    layer.has_previous_state[n]=meta['has_previous_state'][str(n)]
        if values.keys()!=expected or any(not torch.isfinite(t).all()for t in values.values()):raise ValueError('Invalid native payload membership/numerics')
        return cls(cache,m['seen'],m['window'],binding)

def advance(body,capsule,ids,*,binding,visit=None,sites=()):
    """Return hidden and a NEW committed candidate; the input capsule is untouched.

    For a query, discard the returned capsule. For an observed operation, commit
    it with the corresponding public operation/state only after both succeed.
    """
    if torch.is_grad_enabled()or body.training:raise ValueError('Native capsule currently requires eval/no-grad; gradients are a separate contract')
    if binding!=capsule.encoder_binding:raise ValueError('Native experience belongs to a different encoder')
    decoder=body.decoder
    if type(decoder).__name__!='Qwen3_5TextModel':raise ValueError('Reviewed Qwen3.5 native decoder required')
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1 or ids.dtype!=torch.long:raise ValueError('One nonempty unpadded token stream required')
    sites=tuple(sites)
    if sites!=tuple(sorted(set(sites)))or any(type(i)is not int or not 0<=i<len(decoder.layers)for i in sites):raise ValueError('Invalid native core visit sites')
    if (visit is None)!= (not sites):raise ValueError('Explicit visit and sites must be supplied together')
    cache=copy.deepcopy(capsule.cache)
    hidden=_advance_in_place(body,cache,body.embed(ids),seen=capsule.seen,window=capsule.window,visit=visit,sites=sites)
    return hidden,Capsule(cache,capsule.seen+ids.shape[1],capsule.window,binding)

def _advance_in_place(body,cache,hidden,*,seen,window,visit,sites):
    """Owned request scratch only; caller must fork before this mutation."""
    decoder=body.decoder
    if cache.get_seq_length()!=seen:raise ValueError('Native cache/absolute position mismatch')
    positions=(torch.arange(hidden.shape[1],device=hidden.device)+seen)[None]
    first_attention=next(i for i,t in enumerate(decoder.config.layer_types)if t=='full_attention')
    masks=dict(config=decoder.config,inputs_embeds=hidden,attention_mask=None,past_key_values=cache,position_ids=positions)
    attention=create_causal_mask(**masks,layer_idx=first_attention,
        and_mask_function=lambda b,h,q,k:q-k<window)
    recurrent=create_recurrent_attention_mask(**masks)
    rotary=decoder.rotary_emb(hidden,position_ids=positions)
    for i,layer in enumerate(decoder.layers):
        hidden=layer(hidden,position_embeddings=rotary,attention_mask=attention if decoder.config.layer_types[i]=='full_attention'else recurrent,
            position_ids=positions,past_key_values=cache,use_cache=True)
        if not isinstance(hidden,torch.Tensor):raise ValueError('Native layer output contract differs')
        if i in sites:
            transformed=visit(hidden,i)
            if transformed.shape!=hidden.shape:raise ValueError('Native core visit changed shape')
            hidden=transformed
    return decoder.norm(hidden)
