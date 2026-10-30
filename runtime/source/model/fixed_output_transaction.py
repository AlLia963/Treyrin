"""Fixed-encoder output update for a live native session (research contract).

Only vocabulary-output A/B may change. This is not a durable product commit:
the caller must provide one writer and persist weights+four banks as one
generation before claiming crash-safe lifetime learning.
"""
from __future__ import annotations
import copy,hashlib,json
from dataclasses import dataclass,replace
from pathlib import Path
import torch

ALLOWED=('body.output_adapter.A','body.output_adapter.B')

def tensor_sha(t):
    h=hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode())
    raw=t.detach().contiguous().reshape(-1).view(torch.uint8)
    for start in range(0,raw.numel(),8*1024*1024):
        h.update(raw[start:start+8*1024*1024].cpu().numpy().tobytes())
    return h.hexdigest()

def describe(value):
    if isinstance(value,torch.Tensor):return dict(tensor=tensor_sha(value))
    if isinstance(value,dict):return {str(k):describe(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(list,tuple)):return [describe(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,(torch.dtype,torch.device)):return str(value)
    raise TypeError('Unsupported identity value: '+type(value).__name__)

def state_identity(model):
    # state_dict contains serialized NF4 data; explicit live quant_state covers
    # nested absmax/code/offset as well, even if a serializer omits them.
    state={n:tensor_sha(v) for n,v in model.state_dict().items() if n not in ALLOWED}
    quant={}
    for name,p in model.named_parameters():
        q=getattr(p,'quant_state',None)
        if q is not None:
            if not callable(getattr(q,'as_dict',None)):raise ValueError('Opaque quantization state')
            quant[name]=describe(q.as_dict(packed=False))
    topology={name:type(module).__module__+'.'+type(module).__qualname__ for name,module in model.named_modules()}
    hooks={name:dict(pre=[type(x).__qualname__ for x in module._forward_pre_hooks.values()],
        post=[type(x).__qualname__ for x in module._forward_hooks.values()]) for name,module in model.named_modules()
        if module._forward_pre_hooks or module._forward_hooks}
    return dict(state=state,quant=quant,topology=topology,hooks=hooks,
        configuration=model._execution_configuration())

def bank_objects(native):
    return dict(core=native.session.state,native=native.capsule,codes=native.codes,mental=native.mental_codes)

def bank_identity(banks):
    return {name:dict(binding=bank.encoder_binding if name=='native' else bank.binding,tensors={n:tensor_sha(v) for n,v in bank.tensors().items()})
            for name,bank in banks.items()}

@dataclass(frozen=True)
class FixedEncoderCertificate:
    basis:dict
    sources:dict
    binding_before:str

    @classmethod
    def capture(cls,model,sources):
        named=dict(model.named_parameters())
        if any(n not in named or not named[n].requires_grad for n in ALLOWED):raise ValueError('Output-only adapter parameters missing')
        paths={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
        if not paths:raise ValueError('Explicit encoder/runtime source set required')
        return cls(state_identity(model),paths,model._binding)

    def verify(self,model):
        for p,h in self.sources.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Runtime source changed')
        if state_identity(model)!=self.basis:raise ValueError('Frozen encoder/quantization/configuration changed')

def apply_output_update(native,optimizer,values,moments,certificate,*,updates,fault_stage=None):
    """Validate two weights+Adam moments, transport numeric-identical four banks.

    Current query/native output IS ALLOWED to change, unlike a goal-only patch.
    No replay of journal text and no caller-supplied rebind callback is used.
    """
    if fault_stage not in (None,'after_write','after_transport'):raise ValueError('Unknown fault boundary')
    if type(updates) is not int or not 1<=updates<=500:raise ValueError('Bounded update count 1..500 required')
    model=native.candidate.model;native.session._check()
    if model._binding!=certificate.binding_before:raise ValueError('Certificate does not describe current generation')
    certificate.verify(model)
    parameters=dict(model.named_parameters())
    if set(values)!=set(ALLOWED) or set(moments)!=set(ALLOWED):raise ValueError('Exactly two output weights and their moments required')
    membership=[p for group in optimizer.param_groups for p in group['params']]
    if any(sum(p is parameters[n] for p in membership)!=1 for n in ALLOWED):raise ValueError('Optimizer does not own exact output parameters')
    for name in ALLOWED:
        target=parameters[name];v=values[name];m=moments[name]
        if not isinstance(v,torch.Tensor) or v.shape!=target.shape or v.dtype!=target.dtype or not bool(torch.isfinite(v).all()):raise ValueError('Invalid output patch '+name)
        if set(m)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Exact Adam moment schema required')
        step=m['step']
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or not step.is_floating_point() or not bool(torch.isfinite(step).all()) or float(step)<1 or float(step)!=int(float(step)):raise ValueError('Invalid optimizer step')
        previous=optimizer.state.get(target,{}).get('step',0)
        if float(step)!=float(previous)+updates:raise ValueError('Adam step does not match declared updates')
        for key in ('exp_avg','exp_avg_sq'):
            x=m[key]
            if not isinstance(x,torch.Tensor) or x.shape!=target.shape or x.dtype!=target.dtype or not bool(torch.isfinite(x).all()):raise ValueError('Invalid moment '+key)
        if bool((m['exp_avg_sq']<0).any()):raise ValueError('Negative Adam squared moment')
    old_values={n:parameters[n].detach().clone() for n in ALLOWED}
    old_moments={n:copy.deepcopy(optimizer.state.get(parameters[n])) for n in ALLOWED}
    other_moments=describe({str(i):optimizer.state.get(p) for i,p in enumerate(membership) if all(p is not parameters[n] for n in ALLOWED)})
    old_banks=bank_objects(native);before=bank_identity(old_banks);old_steps=native.session.optimizer_steps
    if {x['binding'] for x in before.values()}!={model._binding}:raise ValueError('Four banks differ from current generation')
    def install(banks):
        native.session._set_state(banks['core']);native.candidate.session=native.session
        native.capsule=banks['native'];native.codes=banks['codes'];native.mental_codes=banks['mental']
    try:
        with torch.no_grad():
            for name in ALLOWED:parameters[name].copy_(values[name])
        for name in ALLOWED:
            p=parameters[name];m=moments[name]
            optimizer.state[p]={'step':m['step'].detach().clone().cpu(),
                'exp_avg':m['exp_avg'].detach().clone().to(p.device),
                'exp_avg_sq':m['exp_avg_sq'].detach().clone().to(p.device)}
        if fault_stage=='after_write':raise RuntimeError('Injected after_write')
        certificate.verify(model)
        binding=model.seal_parameters(force_rehash=True)
        from native_stream import Capsule
        capsule=old_banks['native']
        new=dict(core=replace(old_banks['core'],binding=binding),
            native=Capsule(capsule.cache,capsule.seen,capsule.window,binding),
            codes=replace(old_banks['codes'],binding=binding),mental=replace(old_banks['mental'],binding=binding))
        after=bank_identity(new)
        if any(before[n]['tensors']!=after[n]['tensors'] for n in before):raise ValueError('Bank numeric content changed')
        install(new)
        if fault_stage=='after_transport':raise RuntimeError('Injected after_transport')
        native.session._check();native.codes.check(native.session.state);native.mental_codes.check(native.session.state)
        actual=bank_identity(bank_objects(native))
        if actual!=after:raise ValueError('Installed banks differ from validated transport')
        now=describe({str(i):optimizer.state.get(p) for i,p in enumerate(membership) if all(p is not parameters[n] for n in ALLOWED)})
        if now!=other_moments:raise ValueError('Non-output optimizer state changed')
        certificate.verify(model)
        native.session.optimizer_steps=old_steps+updates
        return dict(binding_before=certificate.binding_before,binding_after=binding,
            transported_numeric_identical=True,changed=[n for n in ALLOWED if not torch.equal(old_values[n],parameters[n])],
            quant_states_covered=len(certificate.basis['quant']),query_invariance_required=False,
            crash_safe_product_commit=False,formal_release=False)
    except BaseException:
        with torch.no_grad():
            for name in ALLOWED:parameters[name].copy_(old_values[name])
        for name in ALLOWED:
            if old_moments[name] is None:optimizer.state.pop(parameters[name],None)
            else:optimizer.state[parameters[name]]=old_moments[name]
        restored=model.seal_parameters(force_rehash=True);install(old_banks);native.session.optimizer_steps=old_steps
        if restored!=certificate.binding_before or bank_identity(bank_objects(native))!=before:raise RuntimeError('Rollback identity mismatch')
        certificate.verify(model);native.session._check()
        raise
