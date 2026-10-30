"""Offline research-candidate package/observation/chat entry, not a resident service.

The package includes all body/core parameters, tokenizer, exact body config and
runtime source. It does not need the original model folder after creation.
"""
import argparse
import gc
import hashlib
import inspect
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .structural_core import CoreConfig, ExperienceState
from .structural_model import HFBodyPort, TreyrinStructuralModel


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load_candidate(directory):
    root = Path(directory).resolve()
    package = json.loads((root/'candidate-package.json').read_text(encoding='utf-8'))
    if package.get('schema') != 'treyrin-research-package-0.24-r10':
        raise ValueError('Unsupported research package')
    for name,expected in package['files'].items():
        target = (root/name).resolve()
        if not target.is_relative_to(root) or digest(target) != expected:
            raise ValueError(f'Package asset differs: {name}')
    for name,expected in package['core_source_hashes'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError('Use the reviewed runtime included with this package')
    raw = json.loads((root/'body-config.json').read_text(encoding='utf-8'))
    options = dict(raw)
    model_type = options.pop('model_type')
    config = AutoConfig.for_model(model_type,**options)
    base = AutoModelForCausalLM.from_config(config).float().cpu().eval()
    # Preserve the exact execution-config identity; _name_or_path is metadata,
    # never used to fetch a body. All weights are subsequently loaded locally.
    base.config.update(raw)
    names=package['trainable_body_names']
    if not isinstance(names,list) or len(names)!=len(set(names)):
        raise ValueError('Invalid language-body learning policy')
    model=TreyrinStructuralModel(HFBodyPort(base,package['body_asset_identity'],
        checkpoint_layers=package['checkpoint_language_layers']),
        CoreConfig(**package['core_config']),freeze_body=not bool(names)).eval()
    available=dict(model.body.named_parameters())
    if not set(names)<=set(available):raise ValueError('Unknown trainable body parameters')
    for name,param in available.items():param.requires_grad_(name in names)
    state = model.load_bundle(root/'model')
    tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True,
                                              trust_remote_code=False)
    return model,tokenizer,state,package


def export_candidate(model, tokenizer, state, output, *, training_steps):
    """Export this exact candidate, including trained additions and its state."""
    if type(training_steps) is not int or training_steps < 0:
        raise ValueError('Actual nonnegative optimizer step count required')
    model._check(state, state.working.shape[0])
    sources = ('structural_core.py','structural_model.py','decision_field.py','interleaved_body.py',
               'regional_circuit.py','hypothesis_population.py','hippocampal_circuit.py',
               'r8_circuits.py','qwen_interleaved.py')
    for cls in (type(model), type(model.core)):
        source = Path(inspect.getfile(cls))
        if digest(source) != digest(Path(__file__).parent/source.name):
            raise ValueError('Export requires the exact runtime matching this model')
    source_root = Path(inspect.getfile(type(model))).parent
    for name in sources:
        if digest(source_root/name) != digest(Path(__file__).parent/name):
            raise ValueError('Exported dependency differs from the actual candidate: '+name)
    root = Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    base = model.body.model
    tokenizer.save_pretrained(root/'tokenizer')
    (root/'body-config.json').write_text(json.dumps(base.config.to_dict(),indent=2)+'\n',encoding='utf-8')
    model.export_bundle(root/'model',state)
    runtime = root/'runtime'/'treyrin'
    runtime.mkdir(parents=True)
    (runtime/'__init__.py').write_text('"""Pinned Treyrin research runtime."""\n',encoding='utf-8')
    for name in (*sources,'research_model_cli.py'):
        shutil.copyfile(Path(__file__).parent/name,runtime/name)
    (runtime.parent/'run.py').write_text('from treyrin.research_model_cli import main\nmain()\n',encoding='utf-8')
    package = dict(schema='treyrin-research-package-0.24-r10',
        core_revision='structural-0.24-r10',
        created_at=datetime.now(timezone.utc).isoformat(),training_steps=training_steps,
        body_asset_identity=model.body.asset_identity,core_config=asdict(model.config),
        trainable_body_names=[name for name,p in model.body.named_parameters() if p.requires_grad],
        checkpoint_language_layers=getattr(model.body,'checkpoint_layers',False),
        core_source_hashes={name:digest(runtime/name) for name in sources},
        capability_accepted=False,brain_structure_satisfied=False,files={})
    for path in sorted(root.rglob('*')):
        if path.is_file():
            package['files'][path.relative_to(root).as_posix()] = digest(path)
    (root/'candidate-package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')
    return package


def create_candidate(asset, output):
    asset, root = Path(asset).resolve(),Path(output).resolve()
    tokenizer = AutoTokenizer.from_pretrained(asset,local_files_only=True,trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(asset,local_files_only=True,
        trust_remote_code=False,dtype=torch.float32).cpu().eval()
    identity = {name:digest(asset/name) for name in
                ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json')}
    layers = base.config.num_hidden_layers
    sites = tuple(sorted(set((layers//3-1,2*layers//3-1,layers-1))))
    model = TreyrinStructuralModel(HFBodyPort(base,json.dumps(identity,sort_keys=True)),
        CoreConfig(width=32,episode_slots=16,semantic_slots=4,judgment_mode='separated',
            coupling_mode='layer_interleaved',body_rounds=1,layer_sites=sites,regional_circuit=True,
            hypothesis_mode='adaptive',hippocampal_circuit=True)).eval()
    state = model.initial_state(1)
    export_candidate(model,tokenizer,state,root,training_steps=0)
    ids = tokenizer('Research notes:',return_tensors='pt',add_special_tokens=False).input_ids
    with torch.no_grad():
        reference = model(ids,torch.ones_like(ids),state)
        expected_native = reference.native_logits.clone()
        expected_language = reference.language_logits.clone()
    del reference,model,base,state
    gc.collect()
    # Forbid the pretrained loader during reload: the exported model must stand
    # without fetching/reusing the original model directory or hub weights.
    original_loader = AutoModelForCausalLM.from_pretrained
    def forbidden(*args,**kwargs):
        raise AssertionError('Reload tried to fetch pretrained weights')
    AutoModelForCausalLM.from_pretrained = forbidden
    try:
        restored,_,restored_state,_ = load_candidate(root)
        with torch.no_grad():
            actual = restored(ids,torch.ones_like(ids),restored_state)
        native_difference = (expected_native-actual.native_logits).abs().max().item()
        language_difference = (expected_language-actual.language_logits).abs().max().item()
        assert native_difference == 0 and language_difference == 0, 'Reload output differs'
    finally:
        AutoModelForCausalLM.from_pretrained = original_loader
    report = dict(scope='untrained complete r7 parameter/runtime package; not capability acceptance',
        original_pretrained_loader_forbidden_during_reload=True,
        native_max_difference=native_difference,language_max_difference=language_difference,
        total_package_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        training_steps=0,capability_accepted=False)
    (root.parent/(root.name+'-verification.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def load_experience(path,model):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix+'.json').read_text(encoding='utf-8'))
    if digest(path) != metadata['sha256']:
        raise ValueError('Experience file hash differs')
    data = torch.load(path,map_location='cpu',weights_only=True)
    state = ExperienceState(**data)
    model._check(state,state.working.shape[0])
    return state


def save_experience(path,state):
    path = Path(path)
    meta = path.with_suffix(path.suffix+'.json')
    if path.exists() or meta.exists():
        raise FileExistsError('Use a new state snapshot name')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({**{k:v.detach().cpu() for k,v in state.tensors().items()},'binding':state.binding},path)
    meta.write_text(json.dumps(dict(sha256=digest(path),binding=state.binding,
        created_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    build = sub.add_parser('build')
    build.add_argument('--asset',required=True)
    build.add_argument('--output',required=True)
    for command in ('observe','ask'):
        p = sub.add_parser(command)
        p.add_argument('--package',required=True)
        p.add_argument('--state')
        p.add_argument('--text',required=True)
        if command == 'observe':
            p.add_argument('--source',required=True)
            p.add_argument('--state-output',required=True)
        else:
            p.add_argument('--candidate',action='append',default=[])
            p.add_argument('--max-new-tokens',type=int,default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261024)
    if args.command == 'build':
        print(json.dumps(create_candidate(args.asset,args.output),ensure_ascii=False))
        return
    model,tokenizer,state,package = load_candidate(args.package)
    if args.state:
        state = load_experience(args.state,model)
    ids = tokenizer(args.text,return_tensors='pt',add_special_tokens=False).input_ids
    if args.command == 'observe':
        source = tokenizer(args.source,return_tensors='pt',add_special_tokens=False).input_ids
        with torch.no_grad():
            state = model.ingest(ids,torch.ones_like(ids),source,torch.ones_like(source),state)
        save_experience(args.state_output,state)
        result = dict(state=str(Path(args.state_output).resolve()),writes=state.writes.tolist())
    else:
        kw = {}
        if args.candidate:
            tokens = tokenizer(args.candidate,padding=True,return_tensors='pt',add_special_tokens=False)
            kw = dict(candidate_ids=tokens.input_ids[None],candidate_mask=tokens.attention_mask[None])
        with torch.no_grad():
            judged = model(ids,torch.ones_like(ids),state,**kw)
            generated = model.generate(ids,state,max_new_tokens=args.max_new_tokens,
                                       eos_token_id=tokenizer.eos_token_id,**kw)
        result = dict(text=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True),
            native_logits=judged.native_logits.tolist(),control_logits=judged.control_logits.tolist(),
            predicate_logits=(judged.predicate_logits.tolist() if judged.predicate_logits is not None else None),
            action_values=judged.action_values.tolist(),
            internal_population_gates=(judged.hypothesis_gates.tolist() if judged.hypothesis_gates is not None else None),
            internal_vectors_semantically_validated=False,
            supplied_candidates=args.candidate)
    result.update(research_only=True,training_steps=package['training_steps'],
                  capability_accepted=package['capability_accepted'])
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
