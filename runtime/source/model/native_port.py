"""Explicit native stream execution port, sharing the ordinary Treyrin body."""
from pathlib import Path
import torch
from native_stream import empty_cache,_advance_in_place,digest
from treyrin.structural_model import HFBodyPort

class NativeBodyPort(HFBodyPort):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.runtime_identity.update(native_stream_source_sha256=digest(Path(__file__).with_name('native_stream.py')),
            native_port_source_sha256=digest(__file__))

    def hidden_interleaved_cached(self,embeddings,mask,visit,sites,cache):
        window=getattr(cache,'native_window',None)
        if window is None:return super().hidden_interleaved_cached(embeddings,mask,visit,sites,cache)
        if torch.is_grad_enabled()or self.training or type(window)is not int or window<2:
            raise ValueError('Owned no-grad bounded native stream required')
        if cache.body_cache is None:cache.body_cache=empty_cache(self.decoder.config,window)
        return _advance_in_place(self,cache.body_cache,embeddings,seen=cache.seen_tokens,window=window,visit=visit,sites=sites)
