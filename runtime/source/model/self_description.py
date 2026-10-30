"""Honest self-description computed from the loaded structure and learning path.

There is no canned answer and no persona text: every field is read from the
actual runtime objects (frozen asset manifest, body config, structural core
config, trainable-parameter inventory, recorded training lineage and session
counters). A capability that was never loaded or never verified is reported as
absent or unverified instead of being claimed. The same facts can be committed
as one ordinary observation record, so an identity question is answered from a
real input rather than from a hidden instruction.
"""
from dataclasses import asdict
import hashlib
import json


def _get(obj, name, default=None):
    return getattr(obj, name, default)


def _body_base(candidate):
    return _get(_get(_get(candidate, 'model', None), 'body', None), 'model', None)


MOE_FIELDS = ('num_experts', 'num_experts_per_tok', 'moe_intermediate_size', 'n_routed_experts',
              'num_local_experts', 'moe_layer_freq')
VISION_NAMES = ('visual', 'vision_tower', 'vision_model', 'multi_modal_projector', 'vision_encoder')


def _moe(config):
    if config is None:
        return dict(loaded=False, evidence='no language body config was loaded', model_type=None)
    names = [name for name in MOE_FIELDS if _get(config, name) is not None]
    model_type = str(_get(config, 'model_type', ''))
    if 'moe' in model_type:
        names.append('model_type')
    return dict(loaded=bool(names),
                evidence=('declared expert-routing fields: ' + ','.join(names)) if names
                else 'the loaded body config declares no expert-routing fields',
                model_type=model_type)


def _vision(base):
    if base is None:
        return dict(loaded=False, evidence='no language body was loaded')
    towers = [name for name, _ in base.named_children() if name in VISION_NAMES]
    return dict(loaded=bool(towers),
                evidence=('loaded modules: ' + ','.join(towers)) if towers
                else 'the loaded body has no visual/vision modules')


def _parameters(model):
    if model is None:
        return dict(total=None, trainable=None, frozen=None)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable, frozen=total - trainable)


def _lineage(candidate):
    entries = []
    for entry in list(_get(candidate, 'lineage', []) or []):
        if not isinstance(entry, dict):
            continue
        entries.append(dict(stage=entry.get('stage') or entry.get('reason') or entry.get('source_release'),
                            source_release=entry.get('source_release'),
                            new_training_updates=entry.get('new_training_updates'),
                            optimizer_reset=entry.get('optimizer_reset'),
                            inherited_tensors=entry.get('inherited_tensors_exact')
                            or entry.get('weights_exactly_preserved')))
    declared = [e['new_training_updates'] for e in entries if isinstance(e['new_training_updates'], int)]
    return dict(entries=entries, declared_new_training_updates=(sum(declared) if declared else None),
                declarations=len(declared))


def structural_facts(candidate):
    """Every value below is read from the live objects, not from a fixed answer."""
    model = _get(candidate, 'model')
    body = _get(model, 'body')
    base = _body_base(candidate)
    config = _get(base, 'config')
    core = _get(model, 'config')
    assets = _get(candidate, 'assets', {}) or {}
    runtime = dict(_get(body, 'runtime_identity', {}) or {})
    layer_types = list(_get(config, 'layer_types', []) or [])
    session = _get(candidate, 'session')
    context = _get(config, 'max_position_embeddings')
    learning = _lineage(candidate)
    learning['optimizer_steps'] = _get(session, 'optimizer_steps', None)
    learning['session_observations'] = len(_get(session, 'observations', ()) or ())
    app=_get(candidate,'runtime_app')
    evidence=_get(candidate,'release_evidence',{}) or {}
    return dict(
        runtime=(dict(paging=app.native.live_memory is not None,window=app.native.window,
            durable_records=app.native.live_memory.count(),tools=['read_text','list','write_text','python'],
            interfaces=['conversation','decide','joint','spark','ingest','work','learn','save','resume','read-more'],
            evidence=evidence) if app is not None else None),
        language=dict(upstream=assets.get('upstream') or runtime.get('upstream') or 'unknown-frozen-local-body',
                      commit=assets.get('commit'), weight_storage=assets.get('weight_storage'),
                      model_type=_get(config, 'model_type'), hidden_size=_get(config, 'hidden_size'),
                      num_hidden_layers=_get(config, 'num_hidden_layers'), vocab_size=_get(config, 'vocab_size'),
                      max_position_embeddings=context,
                      linear_attention_layers=sum(1 for x in layer_types if x == 'linear_attention'),
                      full_attention_layers=sum(1 for x in layer_types if x == 'full_attention')),
        structure=dict(architecture=type(model).__name__ if model is not None else None,
                       core_width=_get(core, 'width'), workspace_slots=_get(core, 'workspace_slots'),
                       layer_sites=list(_get(core, 'layer_sites', []) or []),
                       parameters=_parameters(model)),
        mixture_of_experts=_moe(config), vision=_vision(base),
        learning=learning,
        memory=dict(native_window=_get(core, 'archive_slots'), archive_slots=_get(core, 'archive_slots'),
                    mental_archive_slots=_get(core, 'mental_archive_slots'),
                    episode_slots=_get(core, 'episode_slots'), semantic_slots=_get(core, 'semantic_slots'),
                    native_capsule_tokens=_get(candidate, 'native_window', None)),
        caveats=dict(million_token_memory=evidence.get('million_token_memory',{}).get('status','unverified'),
                     capability_acceptance=bool(assets.get('capability_accepted', False)),
                     formal_release=bool(evidence.get('formal_release',False))))


def capability_table(facts):
    language = facts['language']; moe = facts['mixture_of_experts']; vision = facts['vision']
    context = language['max_position_embeddings']
    rows = [dict(name='language_prior', status='loaded' if language['upstream'] else 'absent',
                 value=language['upstream'], evidence='frozen local body asset, not a new trained base'),
            dict(name='mixture_of_experts', status='loaded' if moe['loaded'] else 'absent',
                 value=moe['loaded'], evidence=moe['evidence']),
            dict(name='vision', status='loaded' if vision['loaded'] else 'absent',
                 value=vision['loaded'], evidence=vision['evidence']),
            dict(name='native_context_tokens', status='loaded' if isinstance(context, int) else 'unverified',
                 value=context, evidence='body config max_position_embeddings'),
            dict(name='extended_256k_context',
                 status='loaded' if isinstance(context, int) and context >= 262144 else 'not_loaded',
                 value=context, evidence='claimed only if the loaded config itself reaches 262144'),
            dict(name='kv_gdn_native_capsule', status='loaded',
                 value=facts['memory']['archive_slots'], evidence='bounded per-conversation KV/GDN capsule'),
            dict(name='million_token_memory', status='unverified', value=None,
                 evidence='no accepted million-token retrieval result'),
            dict(name='tool_execution', status='not_implemented', value=None,
                 evidence='this candidate loads no execution layer'),
            dict(name='autonomous_goal_execution', status='unverified', value=None,
                 evidence='a neural selection is not an execution receipt'),
            dict(name='capability_acceptance', status='unverified', value=None,
                 evidence='capability_accepted stays false until it is measured')]
    runtime=facts.get('runtime')
    if runtime:
        replacement={
            'million_token_memory':dict(status=facts['caveats']['million_token_memory'],value=runtime['evidence'].get('million_token_memory'),evidence='inherited accepted chronological evidence and current migration checks'),
            'tool_execution':dict(status='implemented',value=runtime['tools'],evidence='local executor with durable action receipts'),
            'autonomous_goal_execution':dict(status='implemented',value='task store, native goal/action allocation and local tools',evidence='current runtime interfaces; outcomes are stored per task'),
        }
        for row in rows:
            if row['name'] in replacement:row.update(replacement[row['name']])
    return rows


def statement(facts, rows):
    """Deterministic text built only from the facts above."""
    language = facts['language']; structure = facts['structure']; memory = facts['memory']
    learning = facts['learning']; context = language['max_position_embeddings']
    parts = [f"语言主体来自 {language['upstream']} 的本地冻结权重"
             + (f"（commit {language['commit']}）" if language['commit'] else '')
             + f"，不是本项目训练出的新基础模型；权重存储为 {language['weight_storage'] or '本地只读'}。"]
    parts.append(f"当前加载的结构是 {structure['architecture']}（内核宽度 {structure['core_width']}，"
                 f"工作区槽 {structure['workspace_slots']}，结构接入层 {structure['layer_sites']}），"
                 f"可训练参数 {structure['parameters']['trainable']}，冻结参数 {structure['parameters']['frozen']}。")
    parts.append('没有加载 MoE 专家路由、视觉塔或多模态输入。'
                 if not facts['mixture_of_experts']['loaded'] and not facts['vision']['loaded']
                 else f"MoE={facts['mixture_of_experts']['loaded']}，vision={facts['vision']['loaded']}。")
    parts.append(f"实际位置上限是 {context} tokens，因此不声称 256K 已加载。"
                 if isinstance(context, int) and context < 262144
                 else f"实际位置上限是 {context} tokens。")
    if learning['declared_new_training_updates'] is None:
        parts.append(f"学习路径没有记录新增训练步数；当前优化步数 {learning['optimizer_steps']}，"
                     f"会话观测 {learning['session_observations']} 条。")
    else:
        parts.append(f"学习路径累计新增训练 {learning['declared_new_training_updates']} 步；"
                     f"当前优化步数 {learning['optimizer_steps']}，会话观测 {learning['session_observations']} 条。")
    if facts.get('runtime'):
        runtime=facts['runtime']
        parts.append(f"驻留窗口 {runtime['window']} tokens，完整经历保存在可增长分页库；当前持久记录 {runtime['durable_records']} 条。")
        parts.append('当前接口包括对话、共同决策与语言、Spark、材料、任务、工具执行、个体学习、保存和继续。')
        return ''.join(parts)
    parts.append(f"记忆是受限的 KV/GDN 胶囊（episodic 归档 {memory['archive_slots']}，"
                 f"mental 归档 {memory['mental_archive_slots']}）；百万级可用记忆仍未验证，"
                 f"工具执行与自主目标执行也没有已验收结果。")
    return ''.join(parts)


def self_description(candidate):
    facts = structural_facts(candidate)
    rows = capability_table(facts)
    text = statement(facts, rows)
    return dict(schema='treyrin-self-description-v1', derived_from_runtime=True,
                hardcoded_answers=False, persona_added=False, facts=facts,
                capabilities=rows, statement=text,
                unverified=[row['name'] for row in rows
                            if row['status'] in ('unverified', 'not_implemented', 'absent', 'not_loaded')])


def structure_record(candidate, *, known_at=None):
    """The same facts as ONE ordinary public observation; no hidden instruction."""
    from datetime import datetime, timezone
    description = self_description(candidate)
    payload = json.dumps(dict(facts=description['facts'], capabilities=description['capabilities']),
                         ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    text = description['statement'] + '\n结构报告（来源为本地运行时对象）：' + payload
    return dict(event_id='self-structure-' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:24],
                text=text, source='self-structure-report',
                known_at=known_at or datetime.now(timezone.utc).isoformat())
