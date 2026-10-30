"""Same-model cortical opportunity evidence and ONE joint control distribution.

The neural body produces cell logits. These are not generated text or an
external router's decisions. A bounded existing control residual contributes
to the same distribution; mechanical legality is applied before normalization.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import torch
from .goal_control import joint_softmax


def effective_opportunity(state, tick):
    """Resolve inherited deadline identically for the neural read and commit."""
    remaining = tick.deadline_remaining
    if remaining is None and not tick.clear_deadline and bool(state.cognitive_context[0, 1] > 0):
        remaining = float(state.deadline_seconds[0]) - tick.elapsed_seconds
    return replace(tick, deadline_remaining=remaining)


def _opportunity_key(tick):
    # Bind every public opportunity field, including explicit deadline clearing.
    fields = asdict(tick)
    for name in ('elapsed_seconds', 'interruption_cost', 'resource', 'deadline_remaining'):
        if fields[name] is not None:
            fields[name] = float(fields[name])
    payload = json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _state_versions(state):
    return tuple((name, id(value), value._version)
                 for name, value in sorted(state.tensors().items()))


@dataclass(frozen=True)
class CorticalOpportunityRead:
    model_id: int
    state_id: int
    binding: str
    logits: torch.Tensor
    cells: tuple
    replayed_addresses: tuple
    opportunity_key: str
    state_versions: tuple
    logits_version: int

    @classmethod
    def from_episode(cls, model, state, episode, cells, replayed_addresses=(), *, tick):
        if episode.model is not model or episode.state is not state:
            raise ValueError('Opportunity evidence must be produced by the current SAME neural model/state')
        raw = episode.output.native_logits
        if raw.shape != (1, len(cells)) or not torch.isfinite(raw).all():
            raise ValueError('One finite native logit for every actual joint cell required')
        rows = model.config.workspace_slots + 1
        grid = raw.new_zeros((1, rows, 5))
        if len(set(cells)) != len(cells):
            raise ValueError('Duplicate physical goal/action cell')
        for index, (row, action) in enumerate(cells):
            if type(row) is not int or type(action) is not int or not 0 <= row < rows or not 0 <= action < 5:
                raise ValueError('Actual physical goal/action coordinates required')
            grid[0, row, action] = raw[0, index]
        active = ((state.goal_ids[0] > 0) & (state.goal_statuses[0] == 0)).nonzero().flatten().tolist()
        expected = {(row, action) for row in [*active, rows-1] for action in range(5)}
        if set(cells) != expected:
            raise ValueError('Cortical opportunity must cover every active goal and the null row')
        return cls(id(model), id(state), model._binding, grid, tuple(cells), tuple(replayed_addresses),
                   _opportunity_key(tick), _state_versions(state), grid._version)

    def validate(self, model, state, tick):
        if self.model_id != id(model) or self.state_id != id(state) or self.binding != model._binding:
            raise ValueError('Stale or different-model cortical opportunity evidence')
        if self.opportunity_key != _opportunity_key(tick):
            raise ValueError('Different cognitive opportunity: native read cannot be reused')
        if self.state_versions != _state_versions(state):
            raise ValueError('State tensors changed after the native opportunity read')
        if self.logits_version != self.logits._version:
            raise ValueError('Opportunity logits changed after the native read')
        if self.logits.shape != (1, model.config.workspace_slots+1, 5) or not torch.isfinite(self.logits).all():
            raise ValueError('Finite aligned joint cortical logits required')


def cortical_joint_policy(cortical, residual, allowed, *, null_row, residual_bound=.5,calibration=None):
    if cortical.shape != residual.shape or cortical.shape != allowed.shape or cortical.ndim != 3:
        raise ValueError('Aligned cortical/control/legality joint grids required')
    if not torch.isfinite(cortical).all() or not torch.isfinite(residual).all():
        raise ValueError('Finite neural control evidence required')
    # Existing metadata/control logits can otherwise be thousands of times
    # larger than semantic deltas. Keep them useful without allowing scale
    # alone to overwrite the same-body evidence. This fixed research bound
    # requires heldout/individual-adaptation tests before promotion.
    if not 0 <= residual_bound <= 2:
        raise ValueError('Reviewed bounded control contribution required')
    logits = cortical.float() + residual_bound * residual.float().tanh()
    if calibration is not None:
        if calibration.shape!=logits.shape or not torch.isfinite(calibration).all():raise ValueError('Finite aligned learned calibration required')
        logits=logits+calibration
    probabilities = joint_softmax(logits, allowed, null_row=null_row, null_action=0)
    index = probabilities.flatten(1).argmax(-1)
    return dict(probabilities=probabilities, row=index//probabilities.shape[-1],
                action=index%probabilities.shape[-1], logits=logits)
