"""Native opportunity -> SAME cortical graph -> one joint policy -> core step.

Goal content is decoded from the model's protected token bank, never journal
payloads. The opportunity prompt and read are hypothetical and do not commit
body time. Chosen operations update only cognitive state; execution receipts
still require an actual tool result in the product layer.
"""
import copy
import json
import torch
from treyrin.cognitive_workspace import ACTIONS, STATUSES
from treyrin.cortical_control import CorticalOpportunityRead, effective_opportunity

ACTION_DESCRIPTIONS = (
    '等待：此刻暂不值得推进或打扰',
    '内部思考：使用已有记录完成尚未进行的分析、比较或校核，不取得新外部观测',
    '发言：提交已有充分依据的结果或按用户约定适时提醒',
    '补充证据：取得缺少的外部观测或向用户请求必要信息',
    '执行：实施已获授权并满足前置条件的环境操作，选择本身不是执行成功',
)


def _versions(native):
    banks = (native.session.state.tensors(), native.capsule.tensors(),
             native.codes.tensors(), native.mental_codes.tensors())
    return tuple((bank_index, name, id(value), value._version)
                 for bank_index, bank in enumerate(banks)
                 for name, value in sorted(bank.items()))


def opportunity_prompt(native, tick):
    state = native.session.state
    effective = effective_opportunity(state, tick)
    native.mental_codes.check(state); native.codes.check(state)
    active = ((state.goal_ids[0] > 0) & (state.goal_statuses[0] == STATUSES.index('active'))).nonzero().flatten().tolist()
    goals = []
    for slot in active:
        ordinal = int(state.goal_ids[0, slot])
        if int(native.mental_codes.goal_ids[0, slot]) != ordinal:
            raise ValueError('Protected goal identity/code binding differs')
        length = int(native.mental_codes.goal_lengths[0, slot])
        if length <= 0:
            raise ValueError('Occupied goal has no model-owned token code')
        tokens = native.mental_codes.goals[0, slot, :length].tolist()
        goals.append(dict(physical_goal_slot=slot, mental_ordinal=ordinal,
                          actual_model_code=native.candidate.tokenizer.decode(tokens, skip_special_tokens=True),
                          acceptance=int(state.goal_accept[0,slot]), can_act=bool(state.goal_can_act[0,slot]),
                          priority=float(state.goal_priority[0,slot]), interruption_cost=float(state.goal_icost[0,slot])))
    cells, choices = [], []
    for slot in [*active, native.candidate.model.config.workspace_slots]:
        goal = '无目标' if slot == native.candidate.model.config.workspace_slots else f'目标槽{slot}'
        for action, description in enumerate(ACTION_DESCRIPTIONS):
            cells.append((slot, action)); choices.append(goal+'；'+description)
    public = dict(now=tick.known_at, elapsed_seconds=tick.elapsed_seconds,
                  interruption_cost=tick.interruption_cost, resource=tick.resource,
                  deadline_remaining=effective.deadline_remaining,
                  clear_deadline=tick.clear_deadline, goals=goals)
    question = ('这是一次真实认知资源分配机会。根据当前模型记忆和以下公开时间、成本、权限及目标码，'
                '选择现在值得推进的一个目标和一个动作。区分来源报告、已知事实、内部假说和完成回执；'
                '不要把缺失观测假定为已知，不要重复已完成操作。'
                '没有需要推进的目标时，仍可进行有价值的内部分析或等待。\n'
                +json.dumps(public, ensure_ascii=False, sort_keys=True))
    return question, choices, cells, goals


def native_cognitive_tick(native, tick, *, force_replay=False, body_feedback=None):
    # Idempotence/conflict must be checked BEFORE spending a neural forward.
    if tick in native.session.cognitive_ticks:
        return native.session.tick(tick)
    if any(t.operation_id == tick.operation_id for t in native.session.cognitive_ticks):
        raise ValueError('Cognitive identity conflicts')
    native.session._check()
    if tick.resource == 0:
        return native.session.tick(tick)
    before = _versions(native)
    question, choices, cells, goals = opportunity_prompt(native, tick)
    episode, selected = native._query_episode(question, choices, replay_budget=4,
                                             force_replay=force_replay,body_feedback=body_feedback,control_opportunity=True)
    if _versions(native) != before:
        raise ValueError('Hypothetical opportunity forward mutated committed neural state')
    read = CorticalOpportunityRead.from_episode(native.candidate.model, native.session.state,
        episode, cells, selected.get('replayed_record_addresses', ()), tick=tick)
    # Commit the complete tick only after neural/core and bank validations.
    branch = copy.copy(native.session)
    result = branch.tick(tick, cortical_read=read)
    native.codes.check(branch.state); native.mental_codes.check(branch.state)
    native.session = branch; native.candidate.session = branch
    result['cortical_opportunity'] = dict(native_cell_logits=episode.output.native_logits[0].float().cpu().tolist(),
        cells=[dict(goal_slot=s, action=ACTIONS[a]) for s,a in cells],
        goal_addresses=[dict(kind='mental', ordinal=g['mental_ordinal'], goal_slot=g['physical_goal_slot']) for g in goals],
        replayed_addresses=list(read.replayed_addresses),
        native_prefix_committed=False, generated_text_parsed=False, execution_completed=False)
    return result
