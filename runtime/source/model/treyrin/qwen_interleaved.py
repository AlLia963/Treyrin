"""Reviewed no-cache Qwen3.5 text-decoder layer execution with in-graph visits.

Qwen3.5 is a hybrid decoder: GatedDeltaNet linear-attention layers interleaved
with full-attention layers (config.layer_types, full_attention_interval=4).
The decoder/attention/delta-net/mask operations remain those of the installed
Transformers implementation; only explicit residual visits are added. This is
one supported body adapter, not proof of learned cross-body transfer. No
hooks, monkey-patching, mutable global context or second model is used.
"""
import torch
from torch import Tensor


def qwen_interleaved(decoder, embeddings, mask, visit, sites, *, checkpoint_layers=False):
    from transformers.masking_utils import (create_causal_mask,
                                            create_recurrent_attention_mask)
    if type(decoder).__name__ != 'Qwen3_5TextModel':
        raise ValueError('Only the locally reviewed Transformers Qwen3.5 text decoder is supported')
    if getattr(decoder, 'gradient_checkpointing', False):
        raise ValueError('Layer-interleaved checkpoint recomputation is not yet supported')
    sites = tuple(sites)
    n = decoder.config.num_hidden_layers
    if (visit is not None and not sites) or (visit is None and sites) or sites != tuple(sorted(set(sites))) or any(type(i) is not int or not 0 <= i < n for i in sites):
        raise ValueError('Distinct increasing in-range layer visit indices required')
    positions = torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0)
    mask_kwargs = dict(config=decoder.config, inputs_embeds=embeddings,
                       attention_mask=mask, past_key_values=None, position_ids=positions)
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs)}
    hidden = embeddings
    rotary = decoder.rotary_emb(hidden, position_ids=positions)
    for index, layer in enumerate(decoder.layers):
        layer_mask = causal_mask_mapping[decoder.config.layer_types[index]]

        def pure_layer(current, bound_layer=layer, bound_mask=layer_mask):
            return bound_layer(current, position_embeddings=rotary,
                               attention_mask=bound_mask, position_ids=positions,
                               past_key_values=None, use_cache=False)
        if checkpoint_layers and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            # Only pure decoder work is recomputed; visit/state callbacks are
            # outside the closure and therefore cannot execute during backward.
            hidden = checkpoint(pure_layer, hidden, use_reentrant=False, preserve_rng_state=True)
        else:
            hidden = pure_layer(hidden)
        if not isinstance(hidden, Tensor):
            raise ValueError('Installed Qwen3.5 layer output contract differs from reviewed version')
        if index in sites:
            transformed = visit(hidden, index)
            if transformed.shape != hidden.shape:
                raise ValueError('Core visit changed language-body shape')
            hidden = transformed
    return decoder.norm(hidden)


def qwen_interleaved_cached(decoder,embeddings,full_mask,visit,sites,cache):
    """Request-local native Qwen KV/GDN cache with all normal core visits.

    This is autoregressive scratch, not permanent experience or a state migration.
    Only an unpadded single eval stream without autograd is supported.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask,create_recurrent_attention_mask
    if type(decoder).__name__!='Qwen3_5TextModel' or torch.is_grad_enabled() or decoder.training:
        raise ValueError('Cached Qwen visits require reviewed eval/no-grad decoder')
    if embeddings.shape[0]!=1 or not bool((full_mask==1).all()):raise ValueError('One unpadded cached stream required')
    sites=tuple(sites)
    if sites!=tuple(sorted(set(sites))) or any(type(i)is not int or not 0<=i<len(decoder.layers) for i in sites):
        raise ValueError('Invalid cached visit sites')
    if cache.body_cache is None:cache.body_cache=DynamicCache(config=decoder.config)
    if cache.body_cache.get_seq_length()!=cache.seen_tokens:raise ValueError('Decoder scratch/token offset mismatch')
    positions=(torch.arange(embeddings.shape[1],device=embeddings.device)+cache.seen_tokens)[None]
    mask_kwargs=dict(config=decoder.config,inputs_embeds=embeddings,attention_mask=full_mask,
        past_key_values=cache.body_cache,position_ids=positions)
    masks={'full_attention':create_causal_mask(**mask_kwargs),
           'linear_attention':create_recurrent_attention_mask(**mask_kwargs)}
    hidden=embeddings;rotary=decoder.rotary_emb(hidden,position_ids=positions)
    for i,layer in enumerate(decoder.layers):
        hidden=layer(hidden,position_embeddings=rotary,attention_mask=masks[decoder.config.layer_types[i]],
            position_ids=positions,past_key_values=cache.body_cache,use_cache=True)
        if not isinstance(hidden,Tensor):raise ValueError('Cached decoder layer contract differs')
        if i in sites:hidden=visit(hidden,i)
    return decoder.norm(hidden)
