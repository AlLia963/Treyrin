"""Explicit named migration for adding a head to a trained candidate.

Adding parameters to a trained model is only safe when the procedure is
mechanical and checkable:

* every old named tensor is found by NAME and keeps its exact shape and values;
* every new named tensor is named in an allowlist before the load;
* a plain strict load must FAIL (so an accidental "rebind whatever matches"
  restore cannot happen), and the sanctioned load must report exactly the
  allowlisted new names as missing;
* optimizer moments are carried by parameter name, so old moments survive and
  new parameters start without moments (fresh, not zeroed history).

The functions below are pure torch/dict operations; no model, no 4B, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


class MigrationError(ValueError):
    def __init__(self, code, message=''):
        super().__init__('%s: %s' % (code, message))
        self.code = code


@dataclass(frozen=True)
class NamedMigrationReport:
    preserved: tuple
    new: tuple
    missing_after_load: tuple
    values_exact: bool
    optimizer_preserved: tuple = ()
    optimizer_fresh: tuple = ()


def plan_named_migration(old_state, module, *, new_keys):
    """Check the old tensors against the new module WITHOUT loading anything."""
    new_keys = set(new_keys)
    current = module.state_dict()
    preserved, new = [], []
    for name, tensor in old_state.items():
        if name not in current:
            raise MigrationError('mig.old-key-missing', name)
        if tuple(tensor.shape) != tuple(current[name].shape):
            raise MigrationError('mig.shape-mismatch', name)
        preserved.append(name)
    for name in current:
        if name not in old_state:
            if name not in new_keys:
                raise MigrationError('mig.unexpected-new-key', name)
            new.append(name)
    undeclared = new_keys - set(new)
    if undeclared:
        raise MigrationError('mig.declared-but-absent', ','.join(sorted(undeclared)))
    return dict(preserved=tuple(sorted(preserved)), new=tuple(sorted(new)),
                new_count=len(new), preserved_count=len(preserved))


def apply_named_migration(module, old_state, *, new_keys):
    """Sanctioned load: strict load must fail, allowlisted load must succeed."""
    new_keys = set(new_keys)
    plan = plan_named_migration(old_state, module, new_keys=new_keys)
    try:
        module.load_state_dict(old_state, strict=True)
    except RuntimeError:
        pass
    else:
        raise MigrationError('mig.strict-load-succeeded',
                             'a strict load that ignores missing keys would hide a silent rebind')
    result = module.load_state_dict(old_state, strict=False)
    missing = tuple(sorted(result.missing_keys))
    if missing != plan['new']:
        raise MigrationError('mig.missing-set', '%r != %r' % (missing, plan['new']))
    if tuple(sorted(result.unexpected_keys)):
        raise MigrationError('mig.unexpected-key', ','.join(sorted(result.unexpected_keys)))
    after = module.state_dict()
    exact = all(bool(torch.equal(after[name], old_state[name])) for name in plan['preserved'])
    if not exact:
        raise MigrationError('mig.value-changed', 'a preserved tensor changed during load')
    return NamedMigrationReport(preserved=plan['preserved'], new=plan['new'],
                                missing_after_load=missing, values_exact=exact)


def check_optimizer_moments(old_moments, module):
    """Old moments stay attached to the same names; new parameters start empty."""
    params = dict(module.named_parameters())
    preserved, fresh = [], []
    for name, param in params.items():
        if name in old_moments:
            saved = old_moments[name]
            for field, value in saved.items():
                if not torch.is_tensor(value):
                    continue
                if value.ndim > 0 and tuple(value.shape) != tuple(param.shape):
                    raise MigrationError('mig.optimizer-shape', '%s.%s' % (name, field))
            preserved.append(name)
        else:
            fresh.append(name)
    unknown = sorted(set(old_moments) - set(params))
    if unknown:
        raise MigrationError('mig.optimizer-orphan', ','.join(unknown))
    return NamedMigrationReport(preserved=(), new=(), missing_after_load=(),
                                values_exact=True,
                                optimizer_preserved=tuple(sorted(preserved)),
                                optimizer_fresh=tuple(sorted(fresh)))


def snapshot_moments(optimizer, named_params):
    """{param name -> {field -> tensor}} keyed by name, never by position."""
    by_id = {id(param): name for name, param in named_params.items()}
    out = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            name = by_id.get(id(param))
            if name is None:
                raise MigrationError('mig.optimizer-unnamed',
                                     'optimizer holds a parameter outside named_params')
            state = optimizer.state.get(param, {})
            if state:
                out[name] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                             for k, v in state.items()}
    return out
