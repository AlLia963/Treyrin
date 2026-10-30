"""Reviewed no-cache Llama layer execution with in-graph core visits.

The decoder/attention/rotary/mask operations remain those of the installed
Transformers Llama implementation. Only explicit residual visits are added.
This is one supported body adapter, not proof of learned cross-body transfer.
No hooks, monkey-patching, mutable global context or second model is used.
"""
import torch
from torch import Tensor


def llama_interleaved(decoder, embeddings, mask, visit, sites,*,checkpoint_layers=False):
    from transformers.masking_utils import create_causal_mask
    if type(decoder).__name__ != 'LlamaModel' or type(decoder).__module__ != 'transformers.models.llama.modeling_llama':
        raise ValueError('Only the locally reviewed Transformers Llama decoder is supported')
    if decoder.gradient_checkpointing:
        raise ValueError('Layer-interleaved checkpoint recomputation is not yet supported')
    sites = tuple(sites)
    n = decoder.config.num_hidden_layers
    if (visit is not None and not sites) or (visit is None and sites) or sites != tuple(sorted(set(sites))) or any(type(i) is not int or not 0 <= i < n for i in sites):
        raise ValueError('Distinct increasing in-range layer visit indices required')
    positions = torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0)
    causal_mask = create_causal_mask(config=decoder.config, inputs_embeds=embeddings,
        attention_mask=mask, past_key_values=None, position_ids=positions)
    hidden = embeddings
    rotary = decoder.rotary_emb(hidden, position_ids=positions)
    for index, layer in enumerate(decoder.layers[:n]):
        def pure_layer(current,bound_layer=layer):
            return bound_layer(current,attention_mask=causal_mask,position_embeddings=rotary,
                position_ids=positions,past_key_values=None,use_cache=False)
        if checkpoint_layers and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            # Only pure decoder work is recomputed; visit/state callbacks are
            # outside the closure and therefore cannot execute during backward.
            hidden=checkpoint(pure_layer,hidden,use_reentrant=False,preserve_rng_state=True)
        else:hidden=pure_layer(hidden)
        if not isinstance(hidden, Tensor):
            raise ValueError('Installed Llama layer output contract differs from reviewed version')
        if index in sites:
            transformed = visit(hidden, index)
            if transformed.shape != hidden.shape:
                raise ValueError('Core visit changed language-body shape')
            hidden = transformed
    return decoder.norm(hidden)
