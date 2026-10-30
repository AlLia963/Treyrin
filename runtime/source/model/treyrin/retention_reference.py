"""Immutable loss-side language reference; never part of model inference."""
import hashlib
import inspect
import json
from pathlib import Path

import torch


class RetentionReference:
    def __init__(self,body):
        if body.training or any(p.requires_grad for p in body.parameters()):
            raise ValueError('Independent frozen eval-mode reference required')
        self.body=body
        self._versions=self._stamp()
        self._configuration=self._configuration_stamp()
        h=hashlib.sha256()
        h.update(Path(inspect.getfile(type(body))).read_bytes())
        config=getattr(getattr(body,'model',body),'config',None)
        h.update(json.dumps(config.to_dict() if config is not None else {},sort_keys=True,default=str).encode())
        h.update(str(getattr(body,'runtime_identity',None)).encode())
        for name,value in (*body.named_parameters(),*body.named_buffers()):
            h.update(str((name,tuple(value.shape),value.dtype)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.identity=h.hexdigest()

    def _stamp(self):
        return tuple((name,id(value),value._version) for name,value in
                     (*self.body.named_parameters(),*self.body.named_buffers()))

    def _configuration_stamp(self):
        config=getattr(getattr(self.body,'model',self.body),'config',None)
        return json.dumps(dict(config=config.to_dict() if config is not None else None,
            runtime=getattr(self.body,'runtime_identity',None),checkpoint=getattr(self.body,'checkpoint_layers',None)),
            sort_keys=True,default=str)

    def check(self,model):
        if self.body.training or self._stamp()!=self._versions or self._configuration_stamp()!=self._configuration or any(p.requires_grad for p in self.body.parameters()):
            raise ValueError('Retention reference was modified')
        if {id(p) for p in self.body.parameters()}&{id(p) for p in model.parameters()}:
            raise ValueError('Student and frozen reference share parameters')
        if self.body.vocab_size!=model.body.vocab_size or self.body.hidden_size!=model.body.hidden_size:
            raise ValueError('Reference vocabulary or hidden space differs')

    @torch.no_grad()
    def logits(self,ids,mask):
        target_device=ids.device
        reference_device=next(self.body.parameters()).device
        ids,mask=ids.to(reference_device),mask.to(reference_device)
        return self.body.logits(self.body.hidden(self.body.embed(ids),mask)).detach().to(target_device)

    @torch.no_grad()
    def hidden(self,ids,mask):
        device=next(self.body.parameters()).device
        return self.body.hidden(self.body.embed(ids.to(device)),mask.to(device)).detach().to(ids.device)


class CachedRetentionReference(RetentionReference):
    """Frozen, declared last-token probes without a second resident backbone.

    Cache before updating the body. This constrains only these neutral probes;
    it is not a full distribution preservation or task quality guarantee.
    """
    def __init__(self,body,probes):
        reference=RetentionReference(body)
        self.vocab_size,self.hidden_size=body.vocab_size,body.hidden_size
        self._cache={}
        for ids,mask in probes:
            if ids.ndim!=2 or ids.shape[0]!=1 or mask.shape!=ids.shape:
                raise ValueError('Declared single-stream reference probes required')
            key=self._key(ids,mask)
            if key in self._cache:raise ValueError('Duplicate reference probe')
            self._cache[key]=reference.logits(ids,mask)[:,-1:].float().cpu().clone()
        if not self._cache:raise ValueError('Nonempty reference probe set required')
        h=hashlib.sha256(('last-token-probe-cache-v1:'+reference.identity).encode())
        for key,value in sorted(self._cache.items()):
            h.update(key.encode());h.update(value.numpy().tobytes())
        self.identity=h.hexdigest()
        self._versions=tuple((k,id(v),v._version) for k,v in sorted(self._cache.items()))

    @staticmethod
    def _key(ids,mask):
        return json.dumps([ids.detach().cpu().tolist(),mask.detach().cpu().tolist()],separators=(',',':'))

    def check(self,model):
        stamp=tuple((k,id(v),v._version) for k,v in sorted(self._cache.items()))
        if stamp!=self._versions:raise ValueError('Cached retention reference was modified')
        if self.vocab_size!=model.body.vocab_size or self.hidden_size!=model.body.hidden_size:
            raise ValueError('Reference vocabulary or hidden space differs')

    def logits(self,ids,mask):
        key=self._key(ids,mask)
        if key not in self._cache:raise ValueError('Probe outside frozen retention cache')
        return self._cache[key].to(ids.device)

    def hidden(self,ids,mask):
        raise ValueError('Last-token retention cache has no hypothesis hidden references')
