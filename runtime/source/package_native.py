"""Package current native operators into a NEW GGUF output; never overwrite state."""
import argparse,sys,json,hashlib
from pathlib import Path
from gguf_tools import repack,key_bytes,key_string

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path.cwd()/'BF16')
    options=parser.parse_args()
    sys.path.insert(0,str(options.model))
    from modeling_treyrin_runtime import install_runtime
    internal=install_runtime(options.model)
    compiled=Path(__file__).parent/'core-export'
    replacements={}
    for key,filename in [
        ('core_program','treyrin-core-auto.onnx'),('initial_inputs','native-inputs.safetensors'),
        ('observe_program','treyrin-observe.onnx'),('consolidate_program','treyrin-consolidate.onnx'),
        ('memory_inputs','memory-inputs.safetensors'),('candidate_program','treyrin-candidates.onnx'),
        ('candidate_inputs','candidate-inputs.safetensors')]:
        name='treyrin.native.'+key;replacements[name]=key_bytes(name,(compiled/filename).read_bytes())
    for key,path in [('lifecycle_source',internal/'native_lifecycle.py'),
                     ('architecture_source',options.model/'modeling_treyrin_runtime.py')]:
        name='treyrin.native.'+key;replacements[name]=key_bytes(name,path.read_bytes())
    manifest={p.relative_to(internal).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in internal.rglob('*.py')}
    name='treyrin.native.lifecycle_manifest';replacements[name]=key_string(name,json.dumps(manifest,sort_keys=True,separators=(',',':')))
    repack(options.source.resolve(),options.output.resolve(),replacements)

if __name__=='__main__':main()
