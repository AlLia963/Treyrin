"""Local, complete-weight loading and saving for Treyrin v5."""
from pathlib import Path
import sys
import json
import hashlib
import importlib
import shutil
import gc
import datetime
import torch
from .modeling_treyrin_runtime import install_runtime


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def activate_runtime(root):
    root=Path(root).resolve()
    runtime=install_runtime(root)
    if not (runtime/'treyrin/structural_model.py').is_file():
        raise FileNotFoundError(f'Missing bundled Treyrin runtime: {runtime}')
    existing=sys.modules.get('treyrin_paths')
    if existing is not None and Path(existing.__file__).resolve().parent != runtime:
        other=Path(existing.__file__).resolve().parent
        # A saved/reloaded echo uses identical runtime code. Different code must
        # be loaded in a fresh process, rather than silently reusing old modules.
        for file in runtime.rglob('*.py'):
            relative=file.relative_to(runtime)
            if not (other/relative).is_file() or sha256(file)!=sha256(other/relative):
                raise RuntimeError('A different Treyrin runtime is already loaded; use a fresh process')
    if str(runtime) not in sys.path: sys.path.insert(0,str(runtime))
    api=importlib.import_module('treyrin_assembly')
    recipe=importlib.import_module('treyrin_recipe')
    return {name:getattr(api,name) for name in [
        'TreyrinStrictLoadError','key_report','base_body_index','trainable_parameters'
    ]} | {'PROBES':recipe.PROBES}


def load_assembly(root, config, kwargs):
    from accelerate import init_empty_weights
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    root=Path(root).resolve()
    from safetensors import safe_open
    metadata_file=root/'model.safetensors'
    with safe_open(metadata_file,framework='pt',device='cpu') as file:
        manifest=json.loads(file.metadata()['treyrin_manifest'])
    verify=kwargs.pop('verify_assets', 'manifest')
    if verify not in ['full','manifest']: raise ValueError('verify_assets must be full or manifest')
    if verify=='full':
        for relative,digest in manifest['file_hashes'].items():
            if sha256(root/relative)!=digest: raise ValueError(f'Package hash mismatch: {relative}')
    kwargs.pop('local_files_only', None)
    kwargs.pop('trust_remote_code', None)
    kwargs.pop('_from_auto',None)
    kwargs.pop('adapter_kwargs',None)
    info=bool(kwargs.pop('output_loading_info',False))
    dtype=kwargs.pop('dtype',kwargs.pop('torch_dtype',None))
    if dtype not in [None,'auto','bfloat16',torch.bfloat16]:
        raise ValueError('Echo preserves the saved mixed BF16/FP32 dtypes; use dtype="auto"')
    device=kwargs.pop('device',None)
    device_map=kwargs.pop('device_map',None)
    if device_map is not None:
        mapped=device_map.get('') if isinstance(device_map,dict) else device_map
        if mapped=='auto': mapped='cuda:0' if torch.cuda.is_available() else 'cpu'
        if mapped==0: mapped='cuda:0'
        if device is not None and str(device)!=str(mapped):raise ValueError('Conflicting device/device_map')
        device=mapped
    device=str(device or 'cpu')
    quantize=kwargs.pop('quantize',None)
    if quantize is not None or kwargs.pop('quantization_config',None) is not None:
        raise ValueError('This release stores complete BF16/FP32 weights; quantization needs a separately verified export')
    kwargs.pop('low_cpu_mem_usage',None)
    required_binding=kwargs.pop('require_binding',None)
    if kwargs: raise TypeError(f'Unsupported loading options: {sorted(kwargs)}')
    api=importlib.import_module('treyrin_assembly')
    paths=importlib.import_module('treyrin_paths')
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
    text_config=Qwen3_5TextConfig(**(config.text_config.to_dict() if hasattr(config.text_config,'to_dict') else config.text_config))
    text_config._attn_implementation='eager'
    # Match the inherited identity metadata, which is part of the graph seal.
    upstream=manifest['source_manifest']['base_body']
    text_config._name_or_path='treyrin-local-qwen3.5-4b@'+upstream['commit']
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=False):
            base=Qwen3_5ForCausalLM(text_config)
        base.eval().requires_grad_(False)
        adapted,lora_report=api.inject_lora(base,config.lora)
        graph,graph_report=api.build_graph(adapted,tokenizer=tokenizer,
            core_config=config.core_config,asset_identity_value=upstream['asset_manifest_sha256'],
            checkpoint_layers=config.checkpoint_layers,output_adapter_rank=config.output_adapter_rank,
            workspace=install_runtime(root))
    finally:
        torch.set_default_dtype(previous)
    expected=set(graph.state_dict())
    with safe_open(root/'model.safetensors',framework='pt',device='cpu') as archive:
        weight_names=set(archive.keys())
    graph_keys={name for name in weight_names if not name.startswith(('memory.','spark_state.'))}
    if expected!=graph_keys:
        raise ValueError(f'Weight layout mismatch: missing={sorted(expected-weight_names)}, unexpected={sorted(weight_names-expected)}')
    actual={};resident_values={};spark_values={}
    for filename in ['model.safetensors']:
        shard=load_file(str(root/filename),device='cpu')
        for name,value in shard.items():
            if name.startswith('memory.'):
                resident_values[name[7:]]=value
                continue
            if name.startswith('spark_state.'):
                spark_values[name[12:]]=value
                continue
            if list(value.shape)!=manifest['tensor_shapes'][name] or str(value.dtype)!=manifest['tensor_dtypes'][name]:
                raise ValueError(f'Weight shape/dtype mismatch: {name}')
        actual.update({n:v for n,v in shard.items() if not n.startswith(("memory.","spark_state."))})
    graph.load_state_dict(actual,strict=True,assign=True)
    # Tied embeddings are explicitly stored under both standard state_dict keys.
    # Restore the shared Parameter after checking equality through package hashes.
    adapted.get_base_model().tie_weights()
    del actual
    for name,parameter in graph.named_parameters():
        parameter.requires_grad_(name in set(manifest['trainable_names']))
    if any(t.device.type=='meta' for t in graph.state_dict().values()):
        raise ValueError('Unmaterialized model tensor after strict weight loading')
    # config.dtype in the inherited body is BF16, while the cognitive modules
    # remain FP32. Never cast the full combined model to BF16.
    binding=graph.seal_parameters(force_rehash=True)
    if required_binding is not None and binding!=required_binding:
        raise ValueError('Requested binding differs from the loaded model')
    ExperienceState=importlib.import_module('treyrin.structural_core').ExperienceState
    template=graph.core.initial_state(1,binding).tensors()
    if set(resident_values)!=set(template):raise ValueError('Integrated memory tensor set differs')
    for name,value in resident_values.items():
        if value.shape!=template[name].shape or value.dtype!=template[name].dtype:
            raise ValueError('Integrated memory shape/dtype differs: '+name)
    resident=ExperienceState(**resident_values,binding=binding)
    graph.core.validate_state(resident,1)
    cpu_binding=binding
    if device!='cpu':
        graph=graph.to(torch.device(device)).eval()
        binding=graph.seal_parameters(force_rehash=True)
        resident=__import__('dataclasses').replace(resident,**{
            k:v.to(torch.device(device)) for k,v in resident.tensors().items()},binding=binding)
    inherited=manifest['source_manifest']['inherited']
    report=dict(schema='treyrin-v5-load-v1',source='unified-safetensors',directory=str(root),
        context=dict(device=device,dtype='mixed-bfloat16-float32',quantize=None),
        binding=binding,binding_expected=cpu_binding,binding_preserved=binding==cpu_binding,
        encoding_sha256=upstream['encoding_sha256'],
        language_lora=lora_report,missing_keys=[],unexpected_keys=[],
        inherited=dict(checkpoint=inherited['source'],manifest_sha256=inherited['candidate_manifest_sha256'],
            parameter_binding=inherited['parameter_binding'],contract_identity=inherited['contract_identity'],
            optimizer_steps=inherited['optimizer_steps'],trainable_sha256=inherited['trainable_sha256'],
            resident_sha256=inherited['resident_sha256'],binding_preserved=False,
            binding_note='v5 retains all delivered parameters and memory tensors in the standard state_dict'))
    assembly=api.TreyrinAssembly(config=config,body=adapted,graph=graph,resident_state=resident,
        tokenizer=tokenizer,binding=binding,base_body_dir=root,
        source=dict(kind='v5',directory=str(root)),load_report=report)
    assembly.saved_spark_state={n:v.to(torch.device(device)) for n,v in spark_values.items()}
    assembly.integrated_manifest=manifest
    report['source_binding_before_integration']=manifest['source_binding']
    report['binding_migration']='Same learned tensors and memory coordinates; model-owned recurrent Spark and framework source integration'
    return assembly,dict(output_loading_info=info,missing_keys=[],unexpected_keys=[],mismatched_keys=[],error_msgs=[])


def save_treyrin_package(model,directory,*,max_shard_size='2GB',**kwargs):
    """Save weights, factual memory and non-factual Spark in one HF state_dict."""
    from safetensors.torch import save_file
    from huggingface_hub import split_torch_state_dict_into_shards
    safe=kwargs.pop('safe_serialization',True)
    kwargs.pop('is_main_process',None)
    if not safe or kwargs:raise TypeError('Unsupported saving options: '+str(sorted(kwargs)))
    target=Path(directory).resolve()
    if target.exists():raise FileExistsError('Refusing to overwrite '+str(target))
    target.mkdir(parents=True)
    config=model.config.to_dict()
    config['auto_map']={'AutoConfig':'configuration_treyrin.TreyrinConfig','AutoModel':'modeling_treyrin.TreyrinForCausalLM','AutoModelForCausalLM':'modeling_treyrin.TreyrinForCausalLM'}
    config['_name_or_path']='Treyrin-v5'
    (target/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    model.tokenizer.save_pretrained(target)
    if getattr(model,'generation_config',None) is not None:model.generation_config.save_pretrained(target)
    state=model.state_dict()
    manifest=dict(model._assembly.integrated_manifest)
    manifest['source_binding']=model.graph.seal_parameters(force_rehash=True)
    manifest['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest['tensor_shapes']={n:list(v.shape) for n,v in state.items() if not n.startswith(('memory.','spark_state.'))}
    manifest['tensor_dtypes']={n:str(v.dtype) for n,v in state.items() if not n.startswith(('memory.','spark_state.'))}
    manifest['file_hashes']={}
    # Per-tensor hashes in an inherited manifest describe its old checkpoint.
    # A saved model may contain subsequent explicit learning updates.
    manifest.pop('tensor_hashes',None)
    values={n:v.detach().cpu().contiguous().clone() for n,v in state.items()}
    save_file(values,str(target/'model.safetensors'),metadata={'format':'pt',
        'treyrin_manifest':json.dumps(manifest,ensure_ascii=False,separators=(',',':'))})
    for code in model._model_root.glob('*.py'):
        shutil.copy2(code,target/code.name)
    for name in ['LICENSE','THIRD_PARTY_NOTICES.md','README.md']:
        source=model._model_root/name
        if source.exists():shutil.copy2(source,target/name)
    return manifest
