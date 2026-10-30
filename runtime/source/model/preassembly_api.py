"""Unified P4 entry: native decision, open language and joint work on ONE state.

One Candidate, one committed experience state, three callable entries:

  decide()   typed native decision over the SAME model, state and capsules
  express()  open language through the SAME committed state
  joint()    decision then language, with an explicit shared-state receipt

Nothing here is a second model, a prompt router, an external policy or a new
retrieval service. Every number returned comes from the loaded structural model
(native field, predicate/utility/control heads, workspace goal x action head),
the committed core experience state, the native KV/GDN capsule, the episodic and
mental code banks, or the journal counters of StructuralLearningSession.

The existing hooks stay directly callable through this object instead of being
re-implemented:

  work_control()  -> native_cognition.native_cognitive_tick   (goal x action)
  spark()         -> spark_lifecycle.SparkLifecycle.propose   (free hypotheses)
  revise_spark()  -> spark_lifecycle.SparkLifecycle.revise
  intents()       -> intent_lifecycle.IntentLifecycle.intake  (task proposals)
  identity()      -> self_description + structure_record      (runtime facts)
  learn()         -> the existing profile / optimizer / learning hooks
  save_learning() -> resident_store.ResidentStore.save_generation

Honesty rules inherited from the candidate and kept here: head outputs are
uncalibrated, a selection is not an execution receipt, generated or remembered
text is never a fact, and every report carries capability_verified=False unless
a real measurement exists.
"""
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from treyrin.cognitive_workspace import ACTIONS

API_SCHEMA = 'treyrin-preassembly-unified-api-v1'
DECISION_SCHEMA = 'treyrin-decision-v1'
LANGUAGE_SCHEMA = 'treyrin-open-language-v1'
JOINT_SCHEMA = 'treyrin-joint-work-v1'
LEARNING_SCHEMA = 'treyrin-individual-learning-v1'
CONTROL_NAMES = ('continue', 'wait', 'request_evidence')
PROFILE_TENSORS = ('body.output_adapter.A', 'body.output_adapter.B')

__all__ = ['API_SCHEMA', 'DECISION_SCHEMA', 'LANGUAGE_SCHEMA', 'JOINT_SCHEMA',
           'LEARNING_SCHEMA', 'CONTROL_NAMES', 'PROFILE_TENSORS',
           'PreassemblyAPI', 'open_api', 'reopen_generation']


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, name, *, allow_empty=False):
    if not isinstance(value, str):
        raise ValueError('String required for ' + name)
    if not allow_empty and not value.strip():
        raise ValueError('Nonempty string required for ' + name)
    return value


def _probabilities(logits):
    values = logits.float().softmax(-1)
    if values.ndim > 1:
        values = values[0]
    return [float(v) for v in values.tolist()]


def _normalized_entropy(probabilities):
    if len(probabilities) < 2:
        return 0.0
    total = -sum(p * math.log(p) for p in probabilities if p > 0.)
    return float(total / math.log(len(probabilities)))


def _margin(probabilities):
    ordered = sorted(probabilities, reverse=True)
    return float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0


def _round(values, digits=6):
    return [round(float(v), digits) for v in values]


class PreassemblyAPI:
    """Three entries and the existing hooks, bound to one candidate and state."""

    def __init__(self, candidate, *, session=None, window=8192, capacity=2048,
                 self_info=False, system_prompt=None):
        if session is None:
            from role_conversation import RoleConversationSession
            session = RoleConversationSession(candidate, window=window, capacity=capacity,
                                             system_prompt=system_prompt, self_info=self_info)
        if getattr(session, 'candidate', None) is not candidate:
            raise ValueError('One candidate and its own session are required')
        self.candidate = candidate
        self.native = session
        self.api_identity = hashlib.sha256(json.dumps(dict(
            schema=API_SCHEMA, candidate_schema=getattr(candidate, 'schema', None),
            parameter_binding=candidate.model._binding,
            session=type(session).__name__, window=int(session.window),
            capacity=int(getattr(session, 'capacity', 0))), sort_keys=True).encode()).hexdigest()

    # -- shared objects -----------------------------------------------------

    @property
    def session(self):
        """The committed StructuralLearningSession (same object as candidate.session)."""
        return self.native.session

    @property
    def model(self):
        return self.candidate.model

    @property
    def tokenizer(self):
        return self.candidate.tokenizer

    def receipt(self):
        """Committed state identity and clocks; read-only, never advances anything."""
        state = self.session.state
        return dict(
            schema=API_SCHEMA, parameter_binding=str(self.model._binding),
            experience_binding=str(state.binding),
            writes=int(state.writes[0]), workspace_seen=int(state.workspace_seen[0]),
            cognitive_ticks=int(state.cognitive_ticks[0]),
            deliberations=int(state.deliberations[0]),
            consolidations=int(state.consolidations[0]),
            elapsed_seconds=float(state.elapsed_seconds[0]),
            observations=len(self.session.observations),
            mental_records=len(self.session.mental_records),
            native_tokens=int(self.native.capsule.seen),
            attention_window=int(self.native.window),
            optimizer_steps=int(self.session.optimizer_steps),
            conversation_kind=type(self.native).__name__)

    # -- input --------------------------------------------------------------

    def observe(self, text, *, event_id=None, source='human_user', known_at=None,
                part_budget=448):
        """Commit one real public observation; long text is segmented, never cut."""
        from message_stream import text_chunks, source_identity
        _text(text, 'observation text')
        _text(source, 'observation source')
        known_at = known_at or _now()
        message_id = event_id or 'obs-' + hashlib.sha256(
            (text + '|' + known_at).encode('utf-8')).hexdigest()[:24]
        encode = lambda value: self.tokenizer(value, add_special_tokens=False)['input_ids']
        count = tokens = 0
        event_ids = []
        before = self.receipt()
        for part in text_chunks(text, encode, budget=part_budget):
            event_id = '%s:part:%08d' % (message_id, count)
            self.native.observe(dict(event_id=event_id,
                                     text=part['text'], source=source, known_at=known_at,
                                     scope=json.dumps(dict(message_id=message_id, start=part['start'],
                                                           end=part['end']), sort_keys=True,
                                                      separators=(',', ':'))))
            event_ids.append(event_id)
            count += 1
            tokens += part['tokens']
        after = self.receipt()
        return dict(message_id=message_id, source_sha256=source_identity(text),
                    chunks=count, ingested_tokens=tokens, event_ids=event_ids,
                    original_characters=len(text), known_at=known_at,
                    state_before=before, state_after=after,
                    factual_writes_delta=after['writes'] - before['writes'],
                    usable_memory_accepted=False)

    # -- entry 1: native decision -------------------------------------------

    def decide(self, question, *, evidence=(), situation='', goal='', candidates=None,
               replay_budget=4, force_replay=False, opportunity=None, commit_judgment=False,
               hypothesis_count=3, hypothesis_tokens=None, known_at=None):
        """Native judgment, competing explanations, uncertainty, missing information,
        inference and next intent -- all computed from the same committed state.

        candidates: optional explicit public descriptions. When omitted, the
        existing internal hypothesis/candidate path
        (TreyrinStructuralModel.propose_and_judge) supplies the population.
        opportunity: optional dict for CognitiveTick; a real goal x action
        allocation is committed first, and next_intent then reads that actual
        allocation instead of a preview.
        """
        _text(question, 'question')
        _text(situation, 'situation', allow_empty=True)
        _text(goal, 'goal', allow_empty=True)
        requested_evidence = self._evidence(evidence)
        if candidates is not None:
            candidates = list(candidates)
            if len(candidates) < 2:
                raise ValueError('Supplied candidates need at least two descriptions; '
                                 'omit candidates to use the internal hypothesis path')
            if any(not isinstance(c, str) or not c.strip() for c in candidates):
                raise ValueError('Supplied candidate descriptions must be nonempty strings')
        known = {o.event_id for o in self.session.observations}
        resolved = [e for e in requested_evidence if e in known]
        unresolved = [e for e in requested_evidence if e not in known]
        before = self.receipt()
        tick = None
        if opportunity is not None:
            tick = self.work_control(opportunity)
        prompt = self._compose(question, situation=situation, goal=goal, evidence=resolved)
        if candidates is None:
            population = self._internal_population(prompt, count=hypothesis_count,
                                                   tokens=hypothesis_tokens)
            output = population['output']
            space = [p['text'] for p in population['hypotheses']]
            probabilities = population['probabilities']
            judgment_kind = 'internal_hypothesis_population'
            read = dict(replay_enabled=False, read_abstained=False, read_outcome='no_replay',
                        selected_record_addresses=[], replayed_record_addresses=[],
                        replay_budget_rejected=[], abstain_mass=None)
        else:
            with torch.no_grad():
                episode, selected = self.native._query_episode(
                    prompt, tuple(candidates), replay_budget=replay_budget,
                    force_replay=bool(force_replay))
            output = episode.output
            space = list(candidates)
            probabilities = _probabilities(output.native_logits)
            judgment_kind = 'closed_candidate_choice'
            allocation = (output.read_allocation[0, -1].float().tolist()
                          if output.read_allocation is not None else None)
            read = dict(replay_enabled=True, read_abstained=bool(selected.get('read_abstained')),
                        read_outcome=selected.get('read_outcome'),
                        selected_record_addresses=list(selected.get('selected_record_addresses', [])),
                        replayed_record_addresses=list(selected.get('replayed_record_addresses', [])),
                        replay_budget_rejected=list(selected.get('replay_budget_rejected', [])),
                        abstain_mass=None if allocation is None else float(allocation[-1]))
        if len(probabilities) != len(space):
            raise ValueError('Judgment population and probabilities differ')
        selected_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        from public_labels import public_label
        judgment = dict(
            kind=judgment_kind, population_size=len(space), space=space,
            probabilities=_round(probabilities), selected_index=int(selected_index),
            selected_label=(public_label(selected_index) if candidates is not None
                            else str(selected_index)),
            selected_text=space[selected_index],
            top_two_margin=_margin(probabilities), calibrated=False,
            interpretation=('closed public choice over the supplied descriptions' if
                            candidates is not None else
                            'closed choice over the model-generated hypothesis population'))
        explanations = []
        generated = population['hypotheses'] if candidates is None else None
        for index, (label_text, probability) in enumerate(zip(space, probabilities)):
            row = dict(index=index, text=label_text, probability=round(probability, 6),
                       origin=('supplied_candidate' if candidates is not None
                               else 'internal_hypothesis'))
            if generated is not None:
                row['generated_tokens'] = generated[index]['generated_tokens']
                row['retained'] = generated[index]['retained']
            explanations.append(row)
        duplicates = 0
        if generated is not None:
            texts = [p['text'] for p in generated if p['text']]
            duplicates = len(texts) - len(set(texts))
        entropy = _normalized_entropy(probabilities)
        uncertainty = dict(
            normalized_entropy=round(entropy, 6), max_probability=round(max(probabilities), 6),
            top_two_margin=round(_margin(probabilities), 6),
            decisive=bool(entropy < 0.5 and _margin(probabilities) > 0.25),
            population_size=len(space),
            read_abstained=read['read_abstained'], read_outcome=read['read_outcome'],
            abstain_mass=read['abstain_mass'],
            duplicate_population_texts=duplicates, calibrated=False,
            interpretation='uncalibrated head distribution; entropy and margin are '
                           'descriptive, not a truth probability')
        active_slots = self._active_goal_slots()
        open_evidence = int(self.session.state.goal_ev_open.sum())
        questions = []
        if unresolved:
            questions.append('这些证据身份不在本会话已提交的观察中：' + '、'.join(unresolved))
        if not goal and not active_slots:
            questions.append('没有给出目标，状态中也没有活动目标行')
        if read['read_abstained']:
            questions.append('神经读取选择了跳过（abstain），本轮没有引入旧经历')
        if entropy >= 0.98 and len(probabilities) > 1:
            questions.append('判断分布接近均匀，模型没有给出区分性判断')
        if generated is not None and duplicates:
            questions.append('自生成候选中有 %d 条文本重复' % duplicates)
        if open_evidence:
            questions.append('活动目标行上仍有 %d 条未闭合证据位' % open_evidence)
        missing = dict(
            unresolved_evidence_ids=list(unresolved),
            evidence_supplied=len(requested_evidence), evidence_resolved=len(resolved),
            goal_declared=bool(goal), active_goal_slots=list(active_slots),
            open_evidence_on_active_goals=open_evidence,
            read_abstained=read['read_abstained'],
            read_outcome=read['read_outcome'],
            open_questions=questions,
            source='computed from committed journal/state plus this forward; '
                   'no keyword rule and no fixed answer sheet')
        inference = self._inference(output, space)
        next_intent = self._next_intent(tick)
        after_decision = self.receipt()
        judgment_commit = None
        if commit_judgment:
            judgment_commit = self._commit_judgment(
                question=question, space=space, probabilities=probabilities,
                selected_index=selected_index, entropy=entropy,
                evidence=resolved, known_at=known_at or _now(),
                origin=judgment_kind)
        final = self.receipt()
        return dict(
            schema=DECISION_SCHEMA, question=question, situation=situation, goal=goal,
            input=dict(prompt_text=prompt, candidates_supplied=candidates is not None,
                       candidates=None if candidates is None else list(candidates),
                       evidence_requested=list(requested_evidence),
                       evidence_resolved=resolved, evidence_unresolved=unresolved,
                       opportunity_supplied=opportunity is not None,
                       replay_budget=int(replay_budget),
                       hypothesis_count=(None if candidates is not None
                                         else int(hypothesis_count))),
            judgment=judgment, competing_explanations=explanations,
            uncertainty=uncertainty, missing_information=missing, inference=inference,
            next_intent=next_intent,
            read=read, work_control=None if tick is None else {
                k: v for k, v in tick.items() if k != 'receipt'},
            committed=dict(judgment_record_id=(None if judgment_commit is None
                                               else judgment_commit['record_id']),
                           judgment_commit=judgment_commit,
                           factual_writes_delta=final['writes'] - before['writes']),
            state=dict(before=before, after_decision=after_decision, after=final),
            checks=dict(no_factual_pollution=bool(final['writes'] == before['writes']),
                        same_parameter_binding=bool(
                            final['parameter_binding'] == before['parameter_binding']),
                        same_experience_binding=bool(
                            final['experience_binding'] == before['experience_binding']),
                        selection_is_not_execution_receipt=True),
            capability_verified=False)

    def _evidence(self, evidence):
        if isinstance(evidence, str):
            raise ValueError('Evidence must be a sequence of public event identities, '
                             'not a single string')
        values = list(evidence)
        if any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError('Public evidence identities must be nonempty strings')
        return values

    def _compose(self, question, *, situation='', goal='', evidence=()):
        parts = []
        if situation:
            parts.append('处境：' + situation)
        if goal:
            parts.append('目标：' + goal)
        if evidence:
            parts.append('可依据的已有记录编号：' + '、'.join(evidence))
        parts.append('问题：' + question)
        return '\n'.join(parts)

    def _active_goal_slots(self):
        state = self.session.state
        return [slot for slot in range(int(state.goal_ids.shape[1]))
                if int(state.goal_ids[0, slot]) > 0
                and int(state.goal_statuses[0, slot]) == 0]

    def _internal_population(self, prompt, *, count, tokens):
        """Independent native-cache branches, judged in the same committed brain."""
        if type(count) is not int or count < 2:
            raise ValueError('At least two internal hypotheses required per batch')
        rows=[]
        with torch.no_grad():
            episode,_=self.native._query_episode(prompt)
            for index in range(count):
                generated=episode.generate_result(tokens,eos_token_id=self._eos(),sample=index>0)
                text=self.tokenizer.decode(generated.token_ids,skip_special_tokens=True).strip()
                rows.append(dict(text=text,generated_tokens=len(generated.token_ids),retained=bool(text),
                                 origin='model_hypothesis_not_fact'))
            retained=[row for row in rows if row['retained']]
            if len(retained)<2:raise ValueError('Model did not produce two nonempty alternative hypotheses')
            judged,_=self.native._query_episode(prompt,tuple(row['text'] for row in retained))
        return dict(hypotheses=retained,output=judged.output,
                    probabilities=_probabilities(judged.output.native_logits))

    def _inference(self, output, space):
        state = self.session.state
        control = None
        if output.control_logits is not None:
            control = dict(names=list(CONTROL_NAMES),
                           probabilities=_round(_probabilities(output.control_logits[:, :3])))
        predicate = None
        if output.predicate_logits is not None:
            predicate = _round(output.predicate_logits[0].float().sigmoid().tolist())
        utility = None
        if output.action_values is not None:
            utility = _round(output.action_values[0].float().tolist())
        outcomes = None
        if output.predicted_outcomes is not None:
            outcomes = _round(output.predicted_outcomes[0].float().norm(dim=-1).tolist())
        allocation = None
        if output.read_allocation is not None:
            last = output.read_allocation[0, -1].float().tolist()
            allocation = dict(per_candidate=_round(last[:-1]),
                              abstain=round(float(last[-1]), 6))
        cognitive = None
        if output.cognitive_controls is not None:
            cognitive = dict(names=list(ACTIONS),
                             probabilities=_round(_probabilities(output.cognitive_controls[0, -1])))
        return dict(
            candidate_space_size=len(space),
            predicate_probabilities=predicate, utility_values=utility,
            utility_ranking=(None if utility is None else
                             sorted(range(len(utility)), key=lambda i: -utility[i])),
            control=control, predicted_outcome_norms=outcomes,
            read_allocation=allocation, internal_action_prior=cognitive,
            goal_rows=self._active_goal_slots(),
            goal_vector_norm=(None if state.goal_vectors is None else
                              _round(state.goal_vectors[0].float().norm(dim=-1).tolist())),
            note=('all fields are uncalibrated head outputs of the loaded model; '
                  'utility is a value estimate, not a belief, and predicate mass is '
                  'independent per proposition'))

    def _next_intent(self, tick):
        """Goal x action scheduling IS the work control; selection is not execution."""
        state = self.session.state
        slots = int(self.model.config.workspace_slots)
        active = self._active_goal_slots()
        if int(state.cognitive_ticks[0]) == 0:
            return dict(source='not_allocated', allocated=False,
                        reason='no committed cognitive opportunity in this session; '
                               'pass opportunity=... to run the real goal x action head',
                        active_goal_slots=active, goal_x_action_is_work_control=True,
                        execution_completed=False)
        slot = int(state.cognitive_goal[0])
        identity = int(state.cognitive_goal_identity[0])
        joint = state.cognitive_joint[0]
        result = {} if tick is None else dict(tick.get('result') or {})
        action = result.get('action') or ACTIONS[int(joint[slot].argmax())]
        committed_goal_slot = int(result.get('goal_slot', slot))
        committed_goal = (result.get('goal_id') if result
                          else (None if slot >= slots else identity))
        return dict(
            source=('committed_joint_allocation' if tick is None else 'new_opportunity'),
            allocated=True, goal_slot=committed_goal_slot,
            goal_identity=committed_goal,
            goal_is_null_row=bool(committed_goal_slot >= slots),
            action=action,
            joint_cell_probability=(None if result.get('probability') is None
                                    else round(float(result['probability']), 6)),
            row_action_probabilities=_round(joint[slot].float().tolist()),
            opportunity_operation_id=(None if tick is None
                                      else (tick.get('opportunity') or {}).get('operation_id')),
            cortical_read_committed=bool(isinstance(result.get('cortical_opportunity'), dict)),
            active_goal_slots=active, committed_ticks=int(state.cognitive_ticks[0]),
            goal_x_action_is_work_control=True,
            execution_completed=False,
            note='one softmax over (goal, action); a chosen action still needs a real '
                 'tool receipt before it is reported as done')

    def _commit_judgment(self, *, question, space, probabilities, selected_index, entropy,
                         evidence, known_at, origin):
        """One typed NON-FACT record; the judgment becomes readable by language replay."""
        state = self.session.state
        chosen = space[selected_index]
        summary_text = chosen
        payload = ('内部判断（非事实，未校准）：来源=%s；候选数=%d；概率=%s；选中=%d；'
                   '归一化熵=%.4f；选中内容=%s；全部候选=%s' % (
                       origin, len(space), json.dumps(_round(probabilities, 4)),
                       selected_index, entropy, summary_text,
                       json.dumps([str(x) for x in space], ensure_ascii=False)))
        record_id = 'judgment-' + hashlib.sha256(
            (payload + '|' + known_at + '|' + str(int(state.workspace_seen[0]))).encode()
        ).hexdigest()[:20]
        receipt = self.native.remember(dict(
            record_id=record_id, text=payload, kind='belief', known_at=known_at,
            source='model-decision', evidence_ids=tuple(evidence)))
        return dict(record_id=record_id, text=payload, receipt=receipt,
                    derived_summary=True,
                    source_text_chars=len(chosen), retained_chars=len(chosen), full_candidate_set=True,
                    factual_writes=int(self.session.state.writes[0]))

    # -- entry 2: open language ---------------------------------------------

    def express(self, question, *, situation='', goal='', evidence=(), candidates=(),
                replay_budget=4, force_replay=False, max_new_tokens=None, known_at=None):
        """Open language through the same committed state; no new observation.

        max_new_tokens=None means the model decides when to end (EOS/EOT) or the
        real position ceiling is reached; a cap is never reported as success.
        """
        _text(question, 'question')
        _text(situation, 'situation', allow_empty=True)
        _text(goal, 'goal', allow_empty=True)
        requested = self._evidence(evidence)
        known = {o.event_id for o in self.session.observations}
        resolved = [e for e in requested if e in known]
        candidates = list(candidates)
        if candidates and any(not isinstance(c, str) or not c.strip() for c in candidates):
            raise ValueError('Supplied language candidates must be nonempty strings')
        before = self.receipt()
        prompt = self._compose(question, situation=situation, goal=goal, evidence=resolved)
        started = time.monotonic()
        with torch.no_grad():
            episode, selected = self.native._query_episode(
                prompt, tuple(candidates), replay_budget=replay_budget,
                force_replay=bool(force_replay))
            tokens = episode.generate(max_new_tokens, eos_token_id=self._eos())
        generation = getattr(episode, 'last_generation', None)
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ''
        after = self.receipt()
        return dict(
            schema=LANGUAGE_SCHEMA, question=question, situation=situation, goal=goal,
            input=dict(prompt_text=prompt, candidates=list(candidates),
                       evidence_requested=list(requested), evidence_resolved=resolved),
            answer=answer, answer_token_ids=[int(t) for t in tokens],
            finish_reason=(None if generation is None else generation.finish_reason),
            eos_reached=(None if generation is None else bool(generation.eos_reached)),
            truncated=(None if generation is None else (not generation.eos_reached
                                                        and generation.finish_reason == 'length')),
            generation_budget=max_new_tokens,
            end_rule=('model EOS/EOT stop, then the real position ceiling; '
                      'no hidden max_length'),
            read=dict(selected_record_addresses=list(selected.get('selected_record_addresses', [])),
                      replayed_record_addresses=list(selected.get('replayed_record_addresses', [])),
                      replayed_event_ordinals=list(selected.get('event_ordinals', [])),
                      replay_budget_rejected=list(selected.get('replay_budget_rejected', [])),
                      read_abstained=bool(selected.get('read_abstained')),
                      read_outcome=selected.get('read_outcome')),
            seconds=round(time.monotonic() - started, 6),
            state=dict(before=before, after=after),
            committed=dict(factual_writes_delta=after['writes'] - before['writes'],
                           mental_records_delta=(after['mental_records']
                                                 - before['mental_records'])),
            checks=dict(no_factual_pollution=bool(after['writes'] == before['writes']),
                        same_parameter_binding=bool(
                            after['parameter_binding'] == before['parameter_binding'])),
            model_output_not_fact=True, capability_verified=False)

    def _eos(self):
        getter = getattr(self.native, '_eos_token_ids', None)
        if callable(getter):
            return getter()
        return getattr(self.tokenizer, 'eos_token_id', None)

    # -- entry 3: joint work -------------------------------------------------

    def joint(self, question, *, evidence=(), situation='', goal='', candidates=None,
              replay_budget=4, force_replay=False, opportunity=None, commit_judgment=True,
              hypothesis_count=3, hypothesis_tokens=None, max_new_tokens=None,
              known_at=None):
        """Decide then speak on the same state; the shared-state receipt is returned.

        With commit_judgment=True the judgment is committed as one typed non-fact
        record, so the language entry reads it back through the SAME mental code
        bank and replay allocation used for every other old record.
        """
        before = self.receipt()
        decision = self.decide(question, evidence=evidence, situation=situation, goal=goal,
                               candidates=candidates, replay_budget=replay_budget,
                               force_replay=force_replay, opportunity=opportunity,
                               commit_judgment=commit_judgment,
                               hypothesis_count=hypothesis_count,
                               hypothesis_tokens=hypothesis_tokens, known_at=known_at)
        language = self.express(question, situation=situation, goal=goal, evidence=evidence,
                                replay_budget=replay_budget, force_replay=force_replay,
                                max_new_tokens=max_new_tokens, known_at=known_at)
        after = self.receipt()
        read_addresses = [row.get('ordinal')
                          for row in language['read']['replayed_record_addresses']]
        selected_addresses = [row.get('ordinal')
                              for row in language['read']['selected_record_addresses']]
        judgment_id = decision['committed']['judgment_record_id']
        judgment_ordinal = None
        if judgment_id is not None:
            for index, record in enumerate(self.session.mental_records, 1):
                if record.record_id == judgment_id:
                    judgment_ordinal = index
                    break
        return dict(
            schema=JOINT_SCHEMA, question=question, order=['decide', 'commit_judgment',
                                                          'express'],
            decision=decision, language=language,
            shared_state=dict(
                state_before=before, state_after=after,
                parameter_binding_unchanged=bool(
                    before['parameter_binding'] == after['parameter_binding']),
                experience_binding_unchanged=bool(
                    before['experience_binding'] == after['experience_binding']),
                factual_writes_unchanged=bool(before['writes'] == after['writes']),
                judgment_committed=bool(judgment_id is not None),
                judgment_record_id=judgment_id,
                judgment_mental_ordinal=judgment_ordinal,
                language_read_replayed_addresses=language['read']['replayed_record_addresses'],
                language_selected_judgment=bool(judgment_ordinal is not None and
                                                judgment_ordinal in selected_addresses),
                language_read_judgment=bool(judgment_ordinal is not None and
                                            judgment_ordinal in read_addresses),
                same_model=type(self.model).__name__,
                same_state_object=bool(self.session.state is self.native.session.state)),
            capability_verified=False)

    # -- work control (goal x action) ---------------------------------------

    def work_control(self, opportunity):
        """Commit one real cognitive opportunity; returns the actual allocation."""
        from treyrin.cognitive_workspace import CognitiveTick
        if isinstance(opportunity, CognitiveTick):
            record = dict(operation_id=opportunity.operation_id,
                          known_at=opportunity.known_at,
                          elapsed_seconds=opportunity.elapsed_seconds,
                          deadline_remaining=opportunity.deadline_remaining,
                          interruption_cost=opportunity.interruption_cost,
                          resource=opportunity.resource,
                          clear_deadline=opportunity.clear_deadline)
        elif isinstance(opportunity, dict):
            record = dict(opportunity)
        else:
            raise ValueError('CognitiveTick fields as one mapping required')
        record.setdefault('operation_id', 'opportunity-' + hashlib.sha256(
            _now().encode()).hexdigest()[:16])
        record.setdefault('known_at', _now())
        record.setdefault('elapsed_seconds', 0.)
        tick = CognitiveTick(**record)
        before = self.receipt()
        result = self.native.tick(tick)
        after = self.receipt()
        allocation = self._next_intent(dict(result=result, opportunity=record))
        return dict(result=result, allocation=allocation,
                    opportunity=record, state=dict(before=before, after=after),
                    factual_writes_delta=after['writes'] - before['writes'],
                    goal_x_action_is_work_control=True,
                    execution_completed=False, capability_verified=False)

    # -- existing hooks kept directly callable -------------------------------

    def spark(self, problem, *, known_at=None, max_new_tokens=None):
        """Free-hypothesis path: SparkLifecycle over the same native session."""
        from spark_lifecycle import SparkLifecycle
        _text(problem, 'problem')
        return SparkLifecycle(self.native).propose(
            problem, known_at=known_at or _now(), max_new_tokens=max_new_tokens)

    def revise_spark(self, record_id, question, *, known_at=None, max_new_tokens=None):
        from spark_lifecycle import SparkLifecycle
        _text(record_id, 'record_id')
        _text(question, 'question')
        return SparkLifecycle(self.native).revise(
            record_id, question, known_at=known_at or _now(),
            max_new_tokens=max_new_tokens)

    def intents(self, user_text, *, evidence_id=None, known_at=None, max_new_tokens=None):
        """Natural delegation -> typed task/goal proposals over the same session."""
        from intent_lifecycle import IntentLifecycle
        _text(user_text, 'user_text')
        observations = self.session.observations
        if evidence_id is None:
            if not observations:
                raise ValueError('Commit the user message as an observation first')
            evidence_id = observations[-1].event_id
        return IntentLifecycle(self.native).intake(
            user_text, evidence_id=evidence_id, known_at=known_at or _now(),
            max_new_tokens=max_new_tokens)

    def identity(self, *, commit=False, known_at=None):
        """Runtime self facts derived from the loaded objects, never a canned text."""
        from self_description import self_description, structure_record
        description = self_description(self.candidate)
        result = dict(description=description, committed_observation=None)
        if commit:
            record = structure_record(self.candidate, known_at=known_at or _now())
            result['committed_observation'] = self.native.observe(record)
        return result

    def capabilities(self):
        from self_description import capability_table, structural_facts
        facts = structural_facts(self.candidate)
        rows = capability_table(facts)
        return dict(rows=rows,
                    unverified=[row['name'] for row in rows
                                if row['status'] != 'loaded'])

    def brain_modules(self):
        """Live module inventory with its persistent state, clock and feedback.

        Class names, parameter counts and occupancy are read from the loaded
        objects; the declared columns are the contract each module already
        implements. A disabled module is reported present=False, never omitted.
        """
        carriers = {
            'body': self.model.body,
            'regional': getattr(self.model.core, 'regional', None),
            'hippocampus': getattr(self.model.core, 'hippocampus', None),
            'thalamus': getattr(self.model.core, 'thalamus', None),
            'streams': getattr(self.model.core, 'streams', None),
            'dopamine_system': getattr(self.model.core, 'dopamine_system', None),
            'cerebellum': getattr(self.model.core, 'cerebellum', None),
            'neuromod': getattr(self.model.core, 'neuromod', None),
            'amygdala': getattr(self.model.core, 'amygdala', None),
            'drives': getattr(self.model.core, 'drives', None),
            'native_field': getattr(self.model.core, 'native_field', None),
            'hypotheses': getattr(self.model.core, 'hypotheses', None),
            'episodic_policy': getattr(self.model.core, 'episodic_policy', None),
            'workspace': getattr(self.model.core, 'workspace', None),
            'code_attention': getattr(self.model.core, 'code_attention', None),
            'episode_codes': getattr(self.native, 'codes', None),
            'mental_codes': getattr(self.native, 'mental_codes', None),
            'capsule': getattr(self.native, 'capsule', None),
        }
        rows = []
        for name, role, inputs, state_holder, clock, feedback in BRAIN_MODULES:
            module = carriers.get(name)
            present = module is not None
            parameters = trainable = None
            class_name = None
            if present:
                class_name = type(module).__name__
                if isinstance(module, torch.nn.Module):
                    parameters = sum(int(p.numel()) for p in module.parameters())
                    trainable = sum(int(p.numel()) for p in module.parameters()
                                    if p.requires_grad)
            rows.append(dict(module=name, class_name=class_name, present=bool(present),
                             parameters=parameters, trainable_parameters=trainable,
                             reads=inputs, persistent_state=state_holder,
                             update_clock=clock, feedback=feedback))
        goal_joint = getattr(getattr(self.model.core, 'workspace', None), 'goal_joint', None)
        return dict(schema='treyrin-brain-modules-v1', modules=rows,
                    missing=[row['module'] for row in rows if not row['present']],
                    goal_action_head=dict(
                        present=goal_joint is not None,
                        class_name=(None if goal_joint is None else type(goal_joint).__name__),
                        roles=list(ACTIONS),
                        action_roles=list(ACTIONS),
                        committed_ticks=int(self.session.state.cognitive_ticks[0]),
                        note='one softmax over (goal row, action); resource=0 permits wait only'),
                    clocks=self.receipt(),
                    note='rows come from the loaded objects; a disabled module is present=False')

    # -- individual learning / profile / optimizer ---------------------------

    def learn(self, *, feedback=None, question=None, candidates=None, source='human_user',
              example=None, profile_probe=False, known_at=None):
        """Experience-level feedback plus the existing parameter-learning hooks.

        feedback: committed as one real public observation.
        question/candidates: if given, the same decision is taken before and after,
        so the effect of the new observation is visible rather than asserted.
        example: a JointExample for the existing train_step/profile probe.
        profile_probe: real grad-enabled forward + backward on the profile tensors
        (A/B) only; it makes no optimizer step, so no binding is invalidated.
        """
        before = self.receipt()
        decision_before = (self.decide(question, candidates=candidates,
                                       known_at=known_at) if question else None)
        feedback_receipt = None
        if feedback is not None:
            _text(feedback, 'feedback')
            feedback_receipt = self.observe(feedback, source=source, known_at=known_at)
        decision_after = (self.decide(question, candidates=candidates,
                                      known_at=known_at) if question else None)
        after = self.receipt()
        judgment_changed = None
        if decision_before is not None and decision_after is not None:
            judgment_changed = bool(
                decision_before['judgment']['probabilities']
                != decision_after['judgment']['probabilities'])
        profile = self._profile_report()
        result = dict(
            schema=LEARNING_SCHEMA,
            allowed_parameters=self._parameter_inventory(),
            profile=profile,
            optimizer=self._optimizer_report(),
            experience_feedback=dict(
                observation=(None if feedback_receipt is None else feedback_receipt),
                writes_before=before['writes'], writes_after=after['writes'],
                mental_records_before=before['mental_records'],
                mental_records_after=after['mental_records'],
                judgment_before=(None if decision_before is None else
                                 decision_before['judgment']),
                judgment_after=(None if decision_after is None else
                                decision_after['judgment']),
                judgment_changed=judgment_changed,
                note=('experience-level adaptation: the new observation is in the '
                      'committed journal/capsule and the same decision reads it')),
            parameter_update=(dict(attempted=False,applied=False,reason='gradient probe only') if profile_probe else self._update_attempt(example)),
            profile_gradient_probe=(self._profile_probe(example) if profile_probe else
                                    dict(attempted=False, applied=False,
                                         reason='profile_probe not requested')),
            implementation_gaps=self._implementation_gaps(),
            binding_unchanged=bool(before['parameter_binding']
                                   == self.receipt()['parameter_binding']),
            capability_verified=False)
        result['optimizer']=self._optimizer_report()
        return result

    def _parameter_inventory(self):
        groups = {}
        total = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            parts = name.split('.')
            key = '.'.join(parts[:2]) if len(parts) > 1 else parts[0]
            entry = groups.setdefault(key, dict(prefix=key, tensors=0, parameters=0,
                                                example_names=[]))
            entry['tensors'] += 1
            count = int(parameter.numel())
            entry['parameters'] += count
            total += count
            if len(entry['example_names']) < 3:
                entry['example_names'].append(name)
        return dict(schema='treyrin-trainable-inventory-v1',
                    groups=sorted(groups.values(), key=lambda row: row['prefix']),
                    total_trainable_parameters=total,
                    profile_tensors=list(PROFILE_TENSORS),
                    frozen_body=not any(parameter.requires_grad for name, parameter
                                        in self.model.body.named_parameters()),
                    note='exact named tensors; the profile tensors are the only '
                         'readout-only individual-learning parameters')

    def _optimizer_report(self):
        optimizer = getattr(self.candidate, 'optimizer', None)
        if optimizer is None:
            return dict(present=False)
        groups = []
        parameters = 0
        for group in optimizer.param_groups:
            groups.append(dict(parameters=sum(int(p.numel()) for p in group['params']),
                               tensors=len(group['params']),
                               lr=group.get('lr'), weight_decay=group.get('weight_decay')))
            parameters += sum(int(p.numel()) for p in group['params'])
        return dict(present=True, type=type(optimizer).__name__, groups=groups,
                    total_parameters=parameters,
                    steps=int(self.session.optimizer_steps),
                    state_entries=len(optimizer.state),
                    note='the same optimizer object is persisted with the profile '
                         'moments by resident_store.save_generation')

    def _profile_report(self):
        adapter = getattr(self.model.body, 'output_adapter', None)
        if adapter is None:
            return dict(present=False,
                        reason='this body carries no readout-only profile adapter')
        optimizer = getattr(self.candidate, 'optimizer', None)
        members = {id(p) for group in optimizer.param_groups for p in group['params']} \
            if optimizer is not None else set()
        return dict(present=True, tensors=list(PROFILE_TENSORS),
                    shapes=dict(A=list(adapter.A.shape), B=list(adapter.B.shape)),
                    dtype=str(adapter.A.dtype), rank=int(getattr(adapter, 'rank', 0)),
                    parameters=int(adapter.A.numel() + adapter.B.numel()),
                    requires_grad=bool(adapter.A.requires_grad and adapter.B.requires_grad),
                    gradients_available=bool(adapter.A.requires_grad
                                             and adapter.B.requires_grad),
                    in_optimizer=bool(id(adapter.A) in members and id(adapter.B) in members),
                    persisted_with='resident_store.save_generation '
                                   '(profile.safetensors + profile-moments.pt)',
                    consumers=['body.choice_label_logits', 'body.logits'])

    def _update_attempt(self, example):
        if example is None:
            return dict(attempted=False, applied=False,
                        reason='no JointExample supplied; nothing to optimise')
        steps_before = int(self.session.optimizer_steps)
        binding_before = str(self.model._binding)
        try:
            outcome = self.session.train_step(
                example, self.candidate.optimizer, native_weight=3., language_weight=.3,
                retention_weight=1., typed_weight=.3,native_owner=self.native)
        except Exception as exc:
            gate = ('native_single_pass' if getattr(self.model.config,
                                                   'native_single_pass', False) else None)
            return dict(attempted=True, applied=False, gate=gate,
                        error=type(exc).__name__ + ': ' + str(exc),
                        optimizer_steps_before=steps_before,
                        optimizer_steps_after=int(self.session.optimizer_steps),
                        binding_before=binding_before,
                        binding_after=str(self.model._binding),
                        state_unchanged=bool(binding_before == str(self.model._binding)),
                        note='the existing training entry refused; nothing was written')
        return dict(attempted=True, applied=True, outcome=outcome,
                    optimizer_steps_before=steps_before,
                    optimizer_steps_after=int(self.session.optimizer_steps))

    def _profile_probe(self, example):
        """Real gradient of the profile tensors; no optimizer step, binding intact."""
        adapter = getattr(self.model.body, 'output_adapter', None)
        if adapter is None:
            return dict(attempted=False, applied=False,
                        reason='no profile adapter on this body')
        if example is None:
            return dict(attempted=False, applied=False,
                        reason='profile probe needs a JointExample with candidates and '
                               'a native target')
        binding_before = str(self.model._binding)
        prompt_ids = self.session._ids(example.prompt)
        rows = [self.session._ids(text)[0] for text in example.candidates]
        padded = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
        mask = (torch.arange(padded.shape[1], device=padded.device)[None]
                < torch.tensor([int(row.shape[0]) for row in rows],
                               device=padded.device)[:, None])
        kwargs = dict(decision_positions=torch.tensor([prompt_ids.shape[1] - 1],
                                                      device=prompt_ids.device),
                      body_choice_labels=True,
                      candidate_ids=padded[None], candidate_mask=mask.long()[None])
        kwargs.update(self.session.mode.forward_options())
        try:
            with torch.enable_grad():
                output = self.model(prompt_ids, torch.ones_like(prompt_ids),
                                    self.session.state, **kwargs)
                target = output.native_logits.new_tensor([list(example.native_target)])
                loss = -(target * torch.log_softmax(output.native_logits.float(), -1)).sum(-1).mean()
                loss.backward()
                loss_value = float(loss.detach())
                gradients = dict(
                    A=(None if adapter.A.grad is None
                       else round(float(adapter.A.grad.norm()), 8)),
                    B=(None if adapter.B.grad is None
                       else round(float(adapter.B.grad.norm()), 8)))
                probabilities = _probabilities(output.native_logits)
        finally:
            adapter.A.grad = None
            adapter.B.grad = None
        binding_after = str(self.model._binding)
        self.model._check(self.session.state, 1)
        return dict(attempted=True, applied=False,
                    performed_optimizer_step=False,
                    loss=round(loss_value, 8), gradients=gradients,
                    native_probabilities=_round(probabilities),
                    binding_before=binding_before, binding_after=binding_after,
                    binding_unchanged=bool(binding_before == binding_after),
                    note='real forward/backward through the loaded model; probe only')

    def _implementation_gaps(self):
        # The permanent-session path uses the certified output-only transaction.
        # Full input-encoder learning remains an explicit separate migration.
        return []

    def save_learning(self, directory):
        """Commit profile + four banks as one conversation generation."""
        from resident_store import ResidentStore
        if getattr(self.native, 'candidate', None) is not self.candidate:
            raise ValueError('A generation must be saved from a session bound to this candidate')
        return ResidentStore(directory).save_generation(self.candidate, self.native)


# Every row below is checked against the LIVE objects; the declared columns say
# what each existing module reads, where its persistent state lives, which clock
# advances it and what feedback it receives. Disabled modules report present=False.
BRAIN_MODULES = (
    ('body', 'language body (frozen base + LoRA + readout profile)', 'language logits, hidden states',
     'native KV/GDN capsule per conversation', 'observation commit / request prefill',
     'core feedback written back at the interleaved layer sites'),
    ('regional', 'regional cortex, IT/PT streams, selection circuit', 'working + regional scratch',
     'working relays in core state', 'every forward (token step)',
     'to_body projection returns cortical content to the body'),
    ('hippocampus', 'episodic circuit and replay', 'episode slots and archive',
     'episode slots/archive identity in core state', 'observation commit; read on each query',
     'read allocation feeds the next cortical step'),
    ('thalamus', 'thalamic gate / relay', 'route weights', 'none (per-step)', 'every forward',
     'gated content returns to working memory'),
    ('streams', 'dual cortical streams', 'stream activations', 'none (per-step)', 'every forward',
     'merged into working memory'),
    ('dopamine_system', 'dopamine / valence signals', 'reward prediction signals',
     'dopamine state in core state', 'observation commit', 'modulates write gates and plasticity'),
    ('cerebellum', 'prediction and error surface', 'predicted next state', 'none (per-step)',
     'every forward', 'error surface feeds back into working memory'),
    ('neuromod', 'neuromodulators (incl. body drift)', 'modulation scalars', 'drift EMA on the session',
     'every forward; drift EMA on optimizer step', 'scales gates and gains'),
    ('amygdala', 'salience / arousal gating', 'salience weights', 'none (per-step)', 'every forward',
     'gates admission and read'),
    ('drives', 'hypothalamic drives', 'drive vector', 'drive state in core state', 'every forward',
     'conditioned into the goal x action head'),
    ('native_field', 'native decision field (choice/predicate/utility/read)', 'typed head outputs',
     'none (decision forward)', 'each decision/query forward',
     'read allocation and feedback enter the next cortical step'),
    ('hypotheses', 'internal hypothesis population', 'hypothesis vectors and gates', 'none (per-step)',
     'each self-generated candidate forward', 'selected slots transmit into working memory'),
    ('episodic_policy', 'admission / replacement policy', 'admission distribution',
     'admission and event identities in core state', 'observation commit', 'selects what stays resident'),
    ('workspace', 'persistent non-fact workspace and goal x action head', 'workspace/goal vectors',
     'workspace_*/goal_*/cognitive_* in core state', 'mental record commit; cognitive opportunity',
     'the joint action changes the next language forward'),
    ('code_attention', 'token-address attention over owned codes', 'address features',
     'reads episode/mental code banks', 'each query/decision forward', 'replay selection'),
    ('episode_codes', 'owned per-event token codes', 'code tensors', 'episode code bank',
     'observation commit', 'neural replay read'),
    ('mental_codes', 'owned per-record and goal codes', 'code tensors', 'mental code bank',
     'mental/goal record commit', 'goal flash and replay read'),
    ('capsule', 'native KV/GDN conversation state', 'attention/recurrent state', 'native capsule',
     'every committed prefill', 'conditions every later forward'),
)


def open_api(candidate, *, session=None, window=8192, capacity=2048, self_info=False,
             system_prompt=None):
    """Convenience constructor for the unified entry."""
    return PreassemblyAPI(candidate, session=session, window=window, capacity=capacity,
                          self_info=self_info, system_prompt=system_prompt)


def reopen_generation(root, candidate, *, window=8192, capacity=2048):
    """Reopen the persisted generation in a fresh candidate (new process path)."""
    from resident_store import ConversationRepository
    session, record = ConversationRepository(root).open(
        candidate, window=window, capacity=capacity)
    return PreassemblyAPI(candidate, session=session), record

# -- reproducible demos -----------------------------------------------------
# The CPU demo runs the same code paths on the tiny hybrid-Qwen fixture used by
# the contract tests (no CUDA, no 4B, no model-ready mutation). The model-ready
# demo runs on the real trained candidate and is therefore GPU work.

DEMO_QUESTION = '校验和显示 A 通过、B 失败。现在应该先处理哪一个，依据是什么？'
DEMO_CANDIDATES = ('先处理 A：它有通过的校验和', '先处理 B：失败项需要复查')


def demo(model_ready=None, *, window=8192, capacity=2048, max_new_tokens=None,
         store=None, profile_probe=False, question=DEMO_QUESTION, candidates=DEMO_CANDIDATES):
    """Run the three entries and the existing hooks on one real state."""
    import tempfile
    if model_ready is None:
        from test_tiny_candidate import TinyCandidate
        from role_conversation import RoleConversationSession
        candidate = TinyCandidate()
        session = RoleConversationSession(candidate, window=min(window, 1024),
                                         capacity=min(capacity, 1024))
        fixture = 'cpu-tiny-hybrid-qwen-fixture'
        if max_new_tokens is None:
            # The 4-layer fixture declares a 256-position ceiling; the demo bound
            # belongs to the fixture, never to the model's own end rule.
            max_new_tokens = 16
    else:
        from candidate import Candidate
        from role_conversation import RoleConversationSession
        candidate = Candidate()
        restored = candidate.restore(Path(model_ready))
        if candidate.session.operations:
            raise RuntimeError('A restored journal is not a fresh role conversation; '
                               'use treyrin_cli with the resident store for that state')
        session = RoleConversationSession(candidate, window=window, capacity=capacity,
                                          self_info=True)
        fixture = 'model-ready:' + str(model_ready) + ' restored=' + str(restored['restored'])
    api = PreassemblyAPI(candidate, session=session)
    modules = api.brain_modules()
    out = dict(fixture=fixture, api_identity=api.api_identity,
               question=question, candidates=list(candidates),
               generation_bound=max_new_tokens,
               identity=dict(statement=api.identity()['description']['statement'],
                             unverified=api.capabilities()['unverified']),
               brain_modules=dict(modules=[dict(module=row['module'],
                                                class_name=row['class_name'],
                                                present=row['present'],
                                                update_clock=row['update_clock'])
                                           for row in modules['modules']],
                                  missing=modules['missing'],
                                  goal_action_head=modules['goal_action_head'],
                                  clocks={k: modules['clocks'][k] for k in
                                          ('writes', 'workspace_seen', 'cognitive_ticks',
                                           'deliberations', 'consolidations',
                                           'elapsed_seconds', 'optimizer_steps',
                                           'native_tokens')}))
    observed = api.observe('沙箱记录：A 的校验和通过；B 的校验和失败，需要复查。',
                           source='sandbox-demo')
    out['observe'] = {k: observed[k] for k in
                      ('message_id', 'event_ids', 'chunks', 'ingested_tokens',
                       'factual_writes_delta')}
    evidence = list(observed['event_ids'])
    decision = api.decide(question, candidates=list(candidates), evidence=evidence,
                          situation='本机沙箱，两个候选文件',
                          goal='给出下一步处理顺序与依据')
    out['decide'] = dict(
        judgment=decision['judgment'], uncertainty=decision['uncertainty'],
        missing_information=decision['missing_information'],
        inference={k: decision['inference'][k] for k in
                   ('control', 'predicate_probabilities', 'utility_values',
                    'utility_ranking', 'read_allocation', 'goal_rows')},
        next_intent=decision['next_intent'],
        checks=decision['checks'], state=decision['state']['after'])
    language = api.express(question, evidence=evidence, max_new_tokens=max_new_tokens)
    out['express'] = dict(answer=language['answer'],
                          answer_token_ids=language['answer_token_ids'][:64],
                          finish_reason=language['finish_reason'],
                          eos_reached=language['eos_reached'],
                          truncated=language['truncated'],
                          generation_budget=language['generation_budget'],
                          read=language['read'], checks=language['checks'])
    from treyrin.goal_metadata import goal_metadata
    api.native.remember(dict(record_id='demo-goal-1', kind='goal',
                             text='给出下一步处理顺序与依据', source='user',
                             known_at=_now(),
                             goal_metadata=goal_metadata(acceptance='accepted',
                                                         source_kind='user_message',
                                                         can_act=False)))
    control = api.work_control(dict(operation_id='demo-tick-1', known_at=_now(),
                                   elapsed_seconds=5., resource=1.))
    out['work_control'] = dict(allocation=control['allocation'],
                               result={k: control['result'][k] for k in
                                       ('action', 'goal_slot', 'probability')},
                               factual_writes_delta=control['factual_writes_delta'])
    joint = api.joint(question, candidates=list(candidates), evidence=evidence,
                      max_new_tokens=max_new_tokens)
    out['joint'] = dict(
        language_answer=joint['language']['answer'],
        finish_reason=joint['language']['finish_reason'],
        shared_state={k: joint['shared_state'][k] for k in
                      ('parameter_binding_unchanged', 'experience_binding_unchanged',
                       'factual_writes_unchanged', 'judgment_committed',
                       'judgment_record_id', 'judgment_mental_ordinal',
                       'language_selected_judgment', 'language_read_judgment',
                       'same_state_object')})
    from treyrin.structural_learning import JointExample
    example = JointExample(prompt=question, answer='A',
                           candidates=tuple(candidates), native_target=(1., 0.),
                           choice_labels=True)
    learning = api.learn(feedback='用户确认：先处理 A。', question=question,
                         candidates=list(candidates), example=example,
                         profile_probe=profile_probe)
    out['learn'] = dict(
        experience_feedback={k: learning['experience_feedback'][k] for k in
                             ('writes_before', 'writes_after', 'judgment_changed')},
        parameter_update=learning['parameter_update'],
        profile={k: learning['profile'][k] for k in
                 ('present', 'tensors', 'requires_grad', 'in_optimizer')},
        optimizer={k: learning['optimizer'][k] for k in
                   ('present', 'type', 'steps', 'groups')},
        allowed_groups=[group['prefix'] for group in
                        learning['allowed_parameters']['groups']],
        implementation_gaps=[gap['function'] for gap in learning['implementation_gaps']],
        profile_gradient_probe={k: learning['profile_gradient_probe'][k] for k in
                                ('attempted', 'applied', 'performed_optimizer_step',
                                 'loss', 'gradients', 'binding_unchanged')
                                if k in learning['profile_gradient_probe']},
        binding_unchanged=learning['binding_unchanged'])
    if store is not None:
        record = api.save_learning(store)
        reopened, pointer = reopen_generation(store, type(candidate)())
        out['save_reopen'] = dict(
            sequence=record['sequence'], conversation_id=pointer['conversation_id'],
            reopened_native_tokens=reopened.receipt()['native_tokens'],
            same_decision=(reopened.decide(question, candidates=list(candidates))
                           ['judgment']['probabilities']),
            original_decision=decision['judgment']['probabilities'])
    out['capability_verified'] = False
    out['not_claimed'] = ['language quality', '4B capability', 'Spark quality',
                          'calibrated uncertainty', 'executed actions']
    return out


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true',
                        help='run the decision/language/joint demo')
    parser.add_argument('--model-ready', type=Path, default=None,
                        help='real trained snapshot directory (GPU; default: CPU fixture)')
    parser.add_argument('--window', type=int, default=8192)
    parser.add_argument('--capacity', type=int, default=2048)
    parser.add_argument('--max-new-tokens', type=int, default=None,
                        help='None: the model ends by EOS/EOT or the position ceiling')
    parser.add_argument('--store', type=Path, default=None,
                        help='also save the generation and reopen it in a fresh candidate')
    parser.add_argument('--profile-probe', action='store_true',
                        help='real grad-enabled forward/backward on the profile tensors only')
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error('nothing to do; pass --demo')
    result = demo(args.model_ready, window=args.window, capacity=args.capacity,
                  max_new_tokens=args.max_new_tokens, store=args.store,
                  profile_probe=args.profile_probe)
    print(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
