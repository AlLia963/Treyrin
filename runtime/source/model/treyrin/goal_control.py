"""D06 integration: ONE learned head that chooses (goal slot, action) jointly.

    X            (B, R, 29)  per-goal features        (goal_time.spec.PER_GOAL_FEATURES)
    C            (B, 9)      context features         (goal_time.spec.CTX_FEATURES)
    goal_vectors (B, R, 32)  same-body goal semantics  (core.goal_vectors), null row zero
    allowed      (B, R, 5)   legality: resource/permission only, never preference

    -> joint probabilities over R x 5 cells, normalised ONCE per sample

R = slots + 1: rows 0..slots-1 are goals, the last row is the synthetic
"no goal" row.  If a sample has no legal cell at all, probability 1 goes to
(null row, wait) instead of producing NaN or a silent uniform table.

Two properties are deliberate and tested:

* The untrained semantic block is exactly zero, so an untrained head is
  bit-identical to one that never saw goal semantics.  Non-zero logits are not
  evidence of anything; only a training record is.
* legality_mask() reads only permission facts (slot occupied, autonomy for
  'act', remaining resource).  It cannot see deadline, urgency, priority or
  attempts, so no "deadline passed -> speak" rule can hide in the mask.

No numpy in the inference path.  dtype defaults to float32 (CUDA FP32), and
float64 is accepted for finite-difference checks.
"""
from __future__ import annotations

import math

import torch
from torch import nn

PER_GOAL_FEATURES = 29
CTX_FEATURES = 9
GOAL_SEMANTIC = 32
ACTIONS = ('wait', 'think', 'speak', 'request_evidence', 'act')
WAIT = ACTIONS.index('wait')
ACT = ACTIONS.index('act')
NEG = -1e9


class HeadContractError(ValueError):
    pass

# ---------------------------------------------------------------------------
# Frozen ABI.  The order is part of the on-disk contract: it is the D06
# goal_time.spec order, and renumbering it would silently reinterpret weights.
# STATUSES/ACCEPTANCE/SOURCE_KINDS reuse the frozen v6 / D06 enumeration order.
# ---------------------------------------------------------------------------
STATUSES = ('active', 'superseded', 'retracted', 'completed')
ACCEPTANCE = ('pending', 'accepted', 'declined')
SOURCE_KINDS = ('user_message', 'device_log', 'document', 'tool_result', 'model_output', 'schedule')
HORIZON_SECONDS = 3600.0
PER_GOAL_NAMES = (
    'is_null', 'is_empty', 'accept_accepted', 'accept_pending', 'accept_declined',
    'priority', 'has_deadline', 'log_remaining', 'overdue', 'urgency', 'log_icost',
    'log_attempts', 'since_act_h', 'ever_acted', 'ev_missing_frac', 'ev_need_any',
    'can_act', 'is_active', 'is_completed', 'is_retracted', 'is_superseded',
    'is_revision', 'src_user_message', 'src_device_log', 'src_document',
    'src_tool_result', 'src_model_output', 'src_schedule', 'bias')
CTX_NAMES = ('log_elapsed', 'has_deadline_any', 'log_min_remaining', 'log_ctx_cost',
             'resource', 'log_total_elapsed', 'active_fraction', 'max_priority_active',
             'pending_fraction')


def _signed_log(value):
    return math.copysign(math.log1p(abs(float(value))) / 10.0, float(value))


def goal_feature_rows(ids, status, accept, source_kind, priority, icost, deadline_at,
                      deadline_mask, attempts, last_action, evidence, ev_ok, ev_open, revision,
                      can_act, now, slots=None, null_row=None):
    """(slots + 1, 29) per-goal feature matrix, same order as D06 spec.

    All arguments after the scalars are 1-D tensors of length slots (one batch
    element).  evidence is (slots, E) bool: the referenced evidence identities;
    ev_ok counts the satisfied ones and ev_open is the model-declared gap.
    Nothing here decides an action; the wall clock only enters as (deadline-now).
    """
    slots = int(ids.numel() if slots is None else slots)
    if null_row is None:
        null_row = slots
    index = {name: position for position, name in enumerate(PER_GOAL_NAMES)}
    X = ids.new_zeros((slots + 1, len(PER_GOAL_NAMES)), dtype=torch.float32)
    float_rows = torch.stack([ids.float(), status.float(), accept.float(), source_kind.float(),
                              priority.float(), icost.float(), deadline_at.float(),
                              deadline_mask.float(), attempts.float(), last_action.float(),
                              revision.float(), can_act.float(), ev_open.float(),
                              ev_ok.float()], 0)
    (ids_f, status_f, accept_f, source_f, priority_f, icost_f, deadline_f, mask_f,
     attempts_f, last_f, revision_f, can_act_f, ev_open_f, ev_ok_f) = float_rows
    occupied = ids > 0
    empty = ~occupied
    X[:slots, index['bias']] = 1.0
    X[null_row, index['is_null']] = 1.0
    X[null_row, index['bias']] = 1.0
    X[:slots, index['is_empty']] = empty.to(X.dtype)
    X[:slots, index['accept_accepted']] = (accept == ACCEPTANCE.index('accepted')).to(X.dtype)
    X[:slots, index['accept_pending']] = (accept == ACCEPTANCE.index('pending')).to(X.dtype)
    X[:slots, index['accept_declined']] = (accept == ACCEPTANCE.index('declined')).to(X.dtype)
    X[:slots, index['priority']] = priority_f
    X[:slots, index['log_icost']] = torch.log1p(icost_f.clamp_min(0.0)) / 5.0
    X[:slots, index['log_attempts']] = torch.log1p(attempts_f.clamp_min(0.0))
    for name, position in (('is_active', STATUSES.index('active')),
                           ('is_completed', STATUSES.index('completed')),
                           ('is_retracted', STATUSES.index('retracted')),
                           ('is_superseded', STATUSES.index('superseded'))):
        X[:slots, index[name]] = (status == position).to(X.dtype)
    X[:slots, index['is_revision']] = (revision > 0).to(X.dtype)
    X[:slots, index['can_act']] = can_act_f
    covered = evidence.bool().sum(-1).float() if evidence is not None else torch.zeros_like(priority_f)
    open_f = ev_open_f.clamp_min(0.0)
    satisfied = torch.minimum(ev_ok_f.clamp_min(0.0), covered)
    X[:slots, index['ev_need_any']] = ((covered + open_f) > 0).to(X.dtype)
    X[:slots, index['ev_missing_frac']] = torch.where(
        covered > 0, 1.0 - satisfied / covered.clamp_min(1.0),
        torch.where(open_f > 0, torch.ones_like(covered), torch.zeros_like(covered)))
    has_deadline = (mask_f > 0) & occupied
    remaining = deadline_at - float(now)
    X[:slots, index['has_deadline']] = has_deadline.to(X.dtype)
    signed = torch.sign(remaining) * torch.log1p(remaining.abs()) / 10.0
    X[:slots, index['log_remaining']] = torch.where(has_deadline, signed.to(X.dtype),
                                                    torch.zeros_like(signed))
    X[:slots, index['overdue']] = (has_deadline & (remaining <= 0)).to(X.dtype)
    urgency = (1.0 - remaining.to(X.dtype) / HORIZON_SECONDS).clamp(0.0, 1.0)
    X[:slots, index['urgency']] = torch.where(has_deadline, urgency, torch.zeros_like(urgency))
    hours = (float(now) - last_action).clamp_min(0.0) / 3600.0
    acted = last_action > 0
    X[:slots, index['ever_acted']] = acted.to(X.dtype)
    X[:slots, index['since_act_h']] = torch.where(acted, (hours.clamp(max=24.0) / 24.0).to(X.dtype),
                                                  torch.zeros_like(hours))
    onehot = torch.nn.functional.one_hot(source_kind.long().clamp(0, len(SOURCE_KINDS) - 1),
                                         len(SOURCE_KINDS)).to(X.dtype)
    offset = index['src_' + SOURCE_KINDS[0]]
    X[:slots, offset:offset + len(SOURCE_KINDS)] = onehot
    # an unoccupied slot carries identity/labels only, never a record's fields
    keep = occupied[:, None].expand(slots, len(PER_GOAL_NAMES)).to(X.dtype).clone()
    keep[:, index['is_empty']] = 1.0
    keep[:, index['bias']] = 1.0
    X[:slots] = X[:slots] * keep
    return X


def context_features(ids, status, accept, priority, deadline_at, deadline_mask, now,
                     elapsed, total_elapsed, ctx_cost, resource, slots=None):
    """(1, 9) context features, same order and formulas as D06 spec."""
    slots = int(ids.numel() if slots is None else slots)
    index = {name: position for position, name in enumerate(CTX_NAMES)}
    C = ids.new_zeros((1, len(CTX_NAMES)), dtype=torch.float32)
    C[0, index['log_elapsed']] = math.log1p(float(elapsed)) / 10.0
    C[0, index['log_ctx_cost']] = math.log1p(float(ctx_cost))
    C[0, index['resource']] = float(resource)
    C[0, index['log_total_elapsed']] = math.log1p(float(total_elapsed)) / 10.0
    active = (ids > 0) & (status == STATUSES.index('active'))
    C[0, index['active_fraction']] = float(active.sum()) / float(slots)
    if bool(active.any()):
        C[0, index['max_priority_active']] = priority[active].max()
        pending = active & (accept == ACCEPTANCE.index('pending'))
        C[0, index['pending_fraction']] = float(pending.sum()) / float(slots)
        masked = active & (deadline_mask > 0)
        C[0, index['has_deadline_any']] = float(bool(masked.any()))
        if bool(masked.any()):
            remaining = deadline_at[masked] - float(now)
            positive = remaining[remaining > 0]
            pick = positive.min() if positive.numel() else remaining.min()
            C[0, index['log_min_remaining']] = _signed_log(float(pick))
    return C



def legality_mask(occupied, *, can_act=None, resource=1.0, act_resource_floor=0.0,
                  evidence_gap=None, require_evidence_gap=False, actions=len(ACTIONS),
                  null_row=None):
    """(B, R) bool -> (B, R, A) legality.  Resource/permission facts only.

    occupied           (B, R) bool, null row must be True
    can_act            (B, R) bool or None (None = no autonomy restriction)
    resource           scalar remaining resource in [0, 1]
    act_resource_floor 'act' needs at least this much resource
    evidence_gap       (B, R) bool; only used when require_evidence_gap=True, and
                       then only gates 'request_evidence' (asking for evidence a
                       goal does not declare has no referent)
    """
    occupied = torch.as_tensor(occupied).bool()
    if occupied.ndim != 2:
        raise HeadContractError('occupied must be (B, R)')
    batch, rows = occupied.shape
    if null_row is None:
        null_row = rows - 1
    if not 0 <= int(null_row) < rows:
        raise HeadContractError('null_row out of range')
    if bool((~occupied[:, null_row]).any()):
        raise HeadContractError('the null row must stay legal')
    if not 0.0 <= float(resource) <= 1.0:
        raise HeadContractError('resource must be in [0, 1]')
    mask = occupied[:, :, None].expand(batch, rows, actions).clone()
    if can_act is not None:
        can_act = torch.as_tensor(can_act).bool()
        if can_act.shape != occupied.shape:
            raise HeadContractError('can_act must align with occupied')
        mask[:, :, ACT] = mask[:, :, ACT] & can_act
        mask[:, null_row, ACT] = mask[:, null_row, ACT] & False
    if float(resource) < float(act_resource_floor):
        mask[:, :, ACT] = False
    if require_evidence_gap:
        if evidence_gap is None:
            raise HeadContractError('require_evidence_gap needs an evidence_gap tensor')
        gap = torch.as_tensor(evidence_gap).bool()
        if gap.shape != occupied.shape:
            raise HeadContractError('evidence_gap must align with occupied')
        want = ACTIONS.index('request_evidence')
        mask[:, :, want] = mask[:, :, want] & gap
    return mask


class GoalActionHead(nn.Module):
    kind = 'joint_goal_action_mlp'

    def __init__(self, slots=8, *, hidden=16, actions=len(ACTIONS),
                 goal_dim=GOAL_SEMANTIC, per_goal=PER_GOAL_FEATURES, ctx=CTX_FEATURES,
                 dtype=torch.float32, device='cpu', seed=0):
        super().__init__()
        self.slots = int(slots)
        self.rows = self.slots + 1
        self.null_row = self.rows - 1
        self.hidden = int(hidden)
        self.actions = int(actions)
        self.goal_dim = int(goal_dim)
        self.per_goal = int(per_goal)
        self.ctx = int(ctx)
        self.input_dim = self.per_goal + self.ctx + self.goal_dim
        generator = torch.Generator().manual_seed(int(seed))
        fan_in = self.per_goal + self.ctx
        weight = torch.randn(self.input_dim, self.hidden, generator=generator, dtype=dtype)
        weight[:fan_in] /= math.sqrt(fan_in)
        # zero semantic block: an untrained goal-semantics channel is a no-op
        weight[fan_in:] = 0.0
        self.W1 = nn.Parameter(weight.to(device))
        self.b1 = nn.Parameter(torch.zeros(self.hidden, dtype=dtype, device=device))
        self.W2 = nn.Parameter((torch.randn(self.hidden, self.actions, generator=generator,
                                            dtype=dtype) / math.sqrt(self.hidden)).to(device))
        self.b2 = nn.Parameter(torch.zeros(self.actions, dtype=dtype, device=device))

    # -- tensor core ------------------------------------------------------
    def logits(self, X, C, goal_vectors=None):
        X = self._check_x(X)
        C = self._check_c(C, X.shape[0])
        if goal_vectors is None:
            goal_vectors = X.new_zeros(X.shape[0], self.rows, self.goal_dim)
        else:
            goal_vectors = torch.as_tensor(goal_vectors)
            if tuple(goal_vectors.shape) != (X.shape[0], self.rows, self.goal_dim):
                raise HeadContractError('goal_vectors must be (B, R, %d)' % self.goal_dim)
            goal_vectors = goal_vectors.to(X.device, X.dtype)
            if bool(goal_vectors[:, self.null_row, :].abs().sum() != 0):
                raise HeadContractError('the null row goal vector must be all zeros')
        joined = torch.cat([X, C[:, None, :].expand(-1, self.rows, -1), goal_vectors], -1)
        return torch.tanh(joined @ self.W1 + self.b1) @ self.W2 + self.b2

    def _check_x(self, X):
        X = torch.as_tensor(X)
        if X.ndim != 3 or X.shape[1] != self.rows or X.shape[2] != self.per_goal:
            raise HeadContractError('X must be (B, %d, %d)' % (self.rows, self.per_goal))
        return X.to(self.W1.device, self.W1.dtype)

    def _check_c(self, C, batch):
        C = torch.as_tensor(C)
        if C.ndim != 2 or C.shape != (batch, self.ctx):
            raise HeadContractError('C must be (B, %d)' % self.ctx)
        return C.to(self.W1.device, self.W1.dtype)

    def forward(self, X, C, goal_vectors=None, allowed=None, *, return_diagnostics=False):
        logits = self.logits(X, C, goal_vectors)
        probs = joint_softmax(logits, allowed, null_row=self.null_row, null_action=WAIT)
        if not return_diagnostics:
            return probs
        mask = _as_mask(allowed, logits, self.rows, self.actions)
        flat = probs.reshape(probs.shape[0], -1)
        choice = flat.argmax(-1)
        row = torch.div(choice, self.actions, rounding_mode='floor')
        action = choice % self.actions
        picked = flat.gather(1, choice[:, None]).squeeze(1)
        return dict(probabilities=probs, row=row, action=action, probability=picked,
                    legal_cells=mask.reshape(mask.shape[0], -1).sum(-1),
                    fallback=mask.reshape(mask.shape[0], -1).sum(-1) == 0,
                    entropy=-(probs.clamp_min(1e-12).log() * probs).sum((-1, -2)))

    # -- offline initialisation from the D06 experiment head ---------------
    def from_slot_head(self, weights):
        """Explicit named migration from goal_time.policy_torch.TorchSlotHead.

        The 29+9 feature block is copied verbatim (same order, same meaning).
        The old head folded the 32-d goal vector into one scalar logit; this
        head feeds all 32 dimensions through the hidden layer, which is a
        different function.  That channel is therefore zero-initialised and the
        dropped scalar is reported, never silently reinterpreted.
        """
        W1 = torch.as_tensor(weights['W1'])
        if tuple(W1.shape) != (self.per_goal + self.ctx, self.hidden):
            raise HeadContractError('old W1 must be (%d, %d)' % (self.per_goal + self.ctx, self.hidden))
        for name in ('b1', 'W2', 'b2'):
            shape = tuple(getattr(self, name).shape)
            if tuple(torch.as_tensor(weights[name]).shape) != shape:
                raise HeadContractError('old %s must be %s' % (name, shape))
        with torch.no_grad():
            self.W1[:self.per_goal + self.ctx].copy_(
                torch.as_tensor(weights['W1'], dtype=self.W1.dtype))
            self.b1.copy_(torch.as_tensor(weights['b1'], dtype=self.b1.dtype))
            self.W2.copy_(torch.as_tensor(weights['W2'], dtype=self.W2.dtype))
            self.b2.copy_(torch.as_tensor(weights['b2'], dtype=self.b2.dtype))
        semantic = weights.get('semantic.weight')
        return dict(copied=['W1[0:%d]' % (self.per_goal + self.ctx), 'b1', 'W2', 'b2'],
                    zero_initialised=['W1[%d:%d]' % (self.per_goal + self.ctx, self.input_dim)],
                    dropped_scalar_semantic=None if semantic is None else 'semantic.weight',
                    note='the old 32->1 scalar semantic path has no equivalent here')


def _as_mask(allowed, logits, rows, actions):
    if allowed is None:
        return torch.ones_like(logits, dtype=torch.bool)
    mask = torch.as_tensor(allowed, device=logits.device).bool()
    if mask.ndim == 2:
        mask = mask[:, :, None].expand(logits.shape[0], rows, actions)
    if tuple(mask.shape) != tuple(logits.shape):
        raise HeadContractError('allowed must broadcast to (B, R, A)')
    return mask


def joint_softmax(logits, allowed=None, *, null_row, null_action=WAIT):
    """One normalisation per sample across legal cells; legal null fallback."""
    if logits.ndim != 3:
        raise HeadContractError('logits must be (B, R, A)')
    rows, actions = logits.shape[1], logits.shape[2]
    mask = _as_mask(allowed, logits, rows, actions)
    masked = torch.where(mask, logits, torch.full_like(logits, NEG))
    weights = torch.exp(masked - masked.reshape(masked.shape[0], -1).max(-1).values[:, None, None]) * mask
    total = weights.reshape(weights.shape[0], -1).sum(-1)
    probs = weights / torch.where(total > 0, total, torch.ones_like(total))[:, None, None]
    empty = total <= 0
    if bool(empty.any()):
        probs = probs.clone()
        probs[empty] = 0.0
        probs[empty, int(null_row), int(null_action)] = 1.0
    return probs
