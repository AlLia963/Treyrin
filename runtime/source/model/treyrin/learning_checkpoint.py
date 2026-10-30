"""Exclusive, hash-verified CPU learning checkpoints for controlled experiments.

Stores named trainable tensors, the local optimizer and CPU/Python RNG. Frozen
pretrained assets are identified by the experiment binding, never duplicated.
This is not a resident memory migration or cross-backbone transfer mechanism.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import uuid

from safetensors.torch import load_file, save_file
import torch


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameters(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def _optimizer_groups(optimizer, parameters):
    names = {id(parameter): name for name, parameter in parameters.items()}
    groups = [[names.get(id(parameter)) for parameter in group['params']]
              for group in optimizer.param_groups]
    flat = [name for group in groups for name in group]
    if None in flat or len(flat) != len(set(flat)) or set(flat) != set(parameters):
        raise ValueError('Optimizer must cover each named trainable parameter exactly once')
    return groups


def save_learning_checkpoint(path, model, optimizer, *, binding, step):
    """Commit a new directory after all payloads are fsynced; never overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not isinstance(binding, str) or not binding or type(step) is not int or step < 0:
        raise ValueError('Verified experiment binding and nonnegative step required')
    parameters = _parameters(model)
    devices={p.device for p in parameters.values()}
    if not parameters or len(devices)!=1 or next(iter(devices)).type not in ('cpu','cuda'):
        raise ValueError('Checkpoint requires named trainable parameters on one CPU/CUDA device')
    groups = _optimizer_groups(optimizer, parameters)
    if any(not torch.isfinite(parameter).all() for parameter in parameters.values()):
        raise ValueError('Cannot checkpoint nonfinite trainable parameters')
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    temporary.mkdir()
    save_file({name: value.detach().cpu().clone().contiguous() for name, value in parameters.items()},
              temporary/'trainable.safetensors')
    torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if next(iter(devices)).type=='cuda' else [],
                'python_rng': random.getstate()}, temporary/'local-training-state.pt')
    files = {}
    for name in ('trainable.safetensors', 'local-training-state.pt'):
        # Windows FlushFileBuffers needs a writable file descriptor.
        with (temporary/name).open('r+b') as handle:
            os.fsync(handle.fileno())
        files[name] = _sha(temporary/name)
    manifest = {'format': 'treyrin-learning-checkpoint-v2', 'binding': binding,
                'device_type':next(iter(devices)).type,
                'step': step, 'optimizer_class': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
                'optimizer_source_sha256':_sha(Path(inspect.getfile(type(optimizer)))),
                'optimizer_groups': groups, 'files': files}
    with (temporary/'manifest.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(manifest, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    # Windows rename refuses an existing destination. No overwrite operation.
    temporary.rename(path)
    return manifest


def load_learning_checkpoint(path, model, optimizer, *, binding):
    """Validate every payload/binding/name before restoring local state.

    Only this experiment's own verified optimizer file is read, with PyTorch's
    weights-only loader. It is not an import path for untrusted checkpoints.
    Experience is rebuilt from public observations after any parameter update.
    """
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text('utf-8'))
    if manifest.get('format') != 'treyrin-learning-checkpoint-v2' or manifest.get('binding') != binding:
        raise ValueError('Checkpoint format or experiment identity differs')
    if type(manifest.get('step')) is not int or manifest['step'] < 0:
        raise ValueError('Invalid checkpoint step')
    parameters = _parameters(model)
    if manifest['optimizer_class'] != type(optimizer).__module__ + '.' + type(optimizer).__qualname__:
        raise ValueError('Optimizer class differs')
    if manifest.get('optimizer_source_sha256')!=_sha(Path(inspect.getfile(type(optimizer)))):
        raise ValueError('Optimizer implementation differs')
    if manifest['optimizer_groups'] != _optimizer_groups(optimizer, parameters):
        raise ValueError('Optimizer parameter membership/order differs')
    if set(manifest['files']) != {'trainable.safetensors', 'local-training-state.pt'}:
        raise ValueError('Unexpected checkpoint payloads')
    for name, expected in manifest['files'].items():
        if _sha(path/name) != expected:
            raise ValueError('Checkpoint payload hash mismatch: ' + name)
    tensors = load_file(path/'trainable.safetensors', device='cpu')
    if set(tensors) != set(parameters):
        raise ValueError('Trainable parameter names differ')
    for name, parameter in parameters.items():
        value = tensors[name]
        if parameter.device.type != manifest['device_type'] or value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise ValueError('Trainable shape/dtype/device differs: ' + name)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite saved trainable parameter: ' + name)
    state = torch.load(path/'local-training-state.pt', map_location='cpu', weights_only=True)
    if set(state) != {'optimizer', 'torch_rng', 'python_rng','cuda_rng'}:
        raise ValueError('Unexpected local training fields')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors[name])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if manifest['device_type']=='cuda':
        if len(state['cuda_rng'])!=torch.cuda.device_count():
            raise ValueError('CUDA device count differs from saved RNG state')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    return manifest['step']
