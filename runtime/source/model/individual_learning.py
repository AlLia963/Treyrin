"""Real readout learning with the existing certified frozen-encoder transaction."""
from __future__ import annotations
import copy,inspect
from pathlib import Path
import torch
from torch.nn import functional as F
from fixed_output_transaction import ALLOWED,FixedEncoderCertificate,apply_output_update

def learn(native,question,preferred,*,steps=1,learning_rate=None,retention_weight=1.):
    if not isinstance(question,str) or not question.strip() or not isinstance(preferred,str) or not preferred.strip():
        raise ValueError('An explicit feedback question and preferred answer are required')
    if type(steps)is not int or not 1<=steps<=500:raise ValueError('An explicit finite learning batch of 1..500 steps required')
    if getattr(native,'pending_user',None) is not None:raise ValueError('Finish or cancel the current answer before applying a parameter update')
    c=native.candidate;model=c.model;adapter=model.body.output_adapter
    if adapter is None:raise ValueError('This candidate has no individual output adapter')
    certificate=FixedEncoderCertificate.capture(model,[Path(__file__),Path(__file__).with_name('native_port.py'),
        Path(__file__).with_name('native_stream.py'),Path(__file__).with_name('role_conversation.py'),
        Path(__file__).with_name('fixed_output_transaction.py'),Path(__file__).parent/'treyrin/output_adaptation.py'])
    targets=c.tokenizer(preferred,add_special_tokens=False)['input_ids']
    eos=native._eos_token_ids() if hasattr(native,'_eos_token_ids') else [c.tokenizer.eos_token_id]
    if eos:targets.append(eos[0])
    with torch.no_grad():
        episode,_=native._query_episode(question)
        hidden=[episode.output.body_hidden[:,-1:].detach()]
        cache=episode.fork()
        for start in range(0,len(targets)-1,64):
            ids=torch.tensor([targets[start:min(start+64,len(targets)-1)]],device=episode.ids.device)
            out=model(ids,torch.ones_like(ids),native.session.state,_decode_cache=cache,**episode.options)
            hidden.append(out.body_hidden.detach())
        hidden=torch.cat(hidden,1)
        base=model.body.model.get_output_embeddings()(hidden).detach().float()
        reference=(base+adapter.delta(hidden).float()).log_softmax(-1).detach()
        # Retain the actual current individual at fixed, independent probe
        # prompts; teacher words stay loss-side and never enter the journal.
        probes=[]
        for text in native.session.retention_texts:
            ep,_=native._query_episode(text)
            h=ep.output.body_hidden[:,-1:].detach()
            b=model.body.model.get_output_embeddings()(h).float().detach()
            probes.append((h,b,(b+adapter.delta(h).float()).log_softmax(-1).detach()))
    parameters=dict(model.named_parameters());local={n:torch.nn.Parameter(parameters[n].detach().clone()) for n in ALLOWED}
    group=next(g for g in c.optimizer.param_groups if any(p is parameters[ALLOWED[0]] for p in g['params']))
    accepted=set(inspect.signature(torch.optim.AdamW).parameters)-{'params'}
    options={k:v for k,v in group.items() if k in accepted}
    options['lr']=group['lr'] if learning_rate is None else learning_rate
    optimizer=torch.optim.AdamW(list(local.values()),**options)
    for n,p in local.items():
        if parameters[n] in c.optimizer.state:optimizer.state[p]=copy.deepcopy(c.optimizer.state[parameters[n]])
    target=torch.tensor(targets,device=hidden.device).reshape(1,-1)
    def delta(h):return F.linear(F.linear(h.float(),local[ALLOWED[0]]),local[ALLOWED[1]])/adapter.rank
    initial=None;final=None;retained=None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits=base+delta(hidden)
        preferred_loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
        preserve=preferred_loss.new_zeros(())
        for h,b,old in probes:
            new=(b+delta(h)).log_softmax(-1)
            preserve=preserve+(old.exp()*(old-new)).sum(-1).mean().clamp_min(0)
        loss=preferred_loss+retention_weight*preserve/max(1,len(probes))
        if not torch.isfinite(loss):raise ValueError('Nonfinite feedback loss; current individual unchanged')
        loss.backward();torch.nn.utils.clip_grad_norm_(list(local.values()),1.,error_if_nonfinite=True)
        optimizer.step()
        if initial is None:initial=float(preferred_loss.detach())
        final=float(preferred_loss.detach());retained=float(preserve.detach())
    moments={n:copy.deepcopy(optimizer.state[p]) for n,p in local.items()}
    outcome=apply_output_update(native,c.optimizer,{n:p.detach() for n,p in local.items()},moments,certificate,updates=steps)
    native.session.rebuild_with_capsule(native,certificate)
    outcome.update(optimizer_steps=native.session.optimizer_steps,steps=steps,preferred_loss_before=initial,
                   preferred_loss_last=final,retention_kl=retained,teacher_in_experience=False,
                   learning_scope='individual vocabulary readout and corresponding native choice rows')
    return outcome
