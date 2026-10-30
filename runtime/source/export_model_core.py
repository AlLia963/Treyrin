"""Export the exact resident cognitive transition for native C++ execution.

The language decoder stays in llama.cpp/GGML. This transition is invoked inline
at the four existing layer sites; it is not an external service or a router.
"""
from pathlib import Path
import sys,json,dataclasses,hashlib,warnings,collections,os
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file,save_file

ROOT=Path(__file__).parent
ECHO=Path(os.environ.get('TREYRIN_MODEL_DIR',str(Path.cwd()/'BF16'))).resolve()
sys.path.insert(0,str(ECHO))
from modeling_treyrin_runtime import install_runtime
sys.path.insert(0,str(install_runtime(ECHO)))
from treyrin.structural_core import StructuralCore,CoreConfig,ExperienceState
from treyrin.hypothesis_population import HypothesisPopulation
from treyrin.joint_context import selected_goal_context
from treyrin.hippocampal_circuit import HippocampalCircuit
import treyrin.structural_core as core_module

# The source count depends only on architectural width, not tensor values.
# ONNX tracing represents shape entries as tensors; make that metadata integer
# explicit while preserving the same top-k/scatter computation.
def trace_sparse_rate(value,fraction):
    count=max(1,round(int(value.shape[-1])*fraction))
    indices=value.topk(count,-1).indices
    return torch.zeros_like(value).scatter(-1,indices,value.gather(-1,indices))
HippocampalCircuit.sparse_rate=staticmethod(trace_sparse_rate)

class DynamicTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,counts):
        return scores.topk(int(counts.max().item()),dim=-1).indices
    @staticmethod
    def symbolic(g,scores,counts):
        # Retain the data-dependent retrieval count in the native operator graph.
        k=g.op('ReduceMax',counts,keepdims_i=1)
        _,indices=g.op('TopK',scores,k,axis_i=-1,largest_i=1,sorted_i=1,outputs=2)
        return indices

def dynamic_sparse_attention(scores,mask,inverse_temperature,top_fraction):
    sharpened=(scores*inverse_temperature).masked_fill(~mask,torch.finfo(scores.dtype).min)
    if isinstance(top_fraction,torch.Tensor):
        counts=(scores.shape[-1]*top_fraction.detach().reshape(-1)).round().long().clamp(min=1,max=scores.shape[-1])
        top=DynamicTopK.apply(sharpened,counts)
        selected=(torch.arange(top.shape[-1],device=scores.device)[None,None,:]<counts[:,None,None]).expand_as(top)
        hard=torch.zeros_like(sharpened,dtype=torch.bool).scatter(-1,top,selected)&mask
    else:
        keep=min(scores.shape[-1],max(1,round(int(scores.shape[-1])*top_fraction)))
        top=sharpened.topk(keep,dim=-1).indices
        hard=torch.zeros_like(sharpened,dtype=torch.bool).scatter(-1,top,True)&mask
    weights=sharpened.masked_fill(~hard,torch.finfo(sharpened.dtype).min).softmax(-1)*hard.to(sharpened.dtype)
    return weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(sharpened.dtype).tiny)

core_module.sparse_attention=dynamic_sparse_attention

config=json.loads((ECHO/'config.json').read_text('utf-8'))
core=StructuralCore(CoreConfig(**config['core_config'])).eval()
weights={};resident={}
with safe_open(ECHO/'model.safetensors',framework='pt',device='cpu') as archive:
    for name in archive.keys():
        if name.startswith('core.'):weights[name[5:]]=archive.get_tensor(name)
        elif name.startswith('memory.'):resident[name[7:]]=archive.get_tensor(name)
core.load_state_dict(weights,strict=True)
core.requires_grad_(False)
state_names=sorted(resident)
scratch_names=['working','cortex','relay','selection','hippocampal_population','cerebellar_prediction','dopamine']
pop_names=['vectors','gates','selection_logits','episode_attention']
surface_names=['shared','native_logits','control_logits','predicted_outcomes','candidate_reads',
    'route_weights','predicate_logits','action_values','read_allocation','regional_edges',
    'hippocampal_mismatch','thalamic','cerebellar_error','neuromod','drive_vector',
    'initiate_logit','cognitive_control_logits']

class Transition(torch.nn.Module):
    def __init__(self):super().__init__();self.core=core
    def forward(self,x,echo_hash,*data):
        scratch=data[:7];pop=data[7:11];values=data[11:]
        state=ExperienceState(**dict(zip(state_names,values)),binding='native-port-input')
        population=HypothesisPopulation(*pop)
        goal=selected_goal_context(state)
        result=self.core.step(x,scratch[0],state,None,goal,
            regional_state=tuple(scratch[1:4]),hippocampal_state=scratch[4],
            cerebellar_prediction_prev=scratch[5],da_prev=scratch[6],
            hypothesis_state=population,echo_query_hash=echo_hash,
            drift=torch.zeros_like(x[:,0]),amygdala_mode='normal')
        return tuple(getattr(result,n) for n in scratch_names)+tuple(
            getattr(result.hypotheses,n) for n in pop_names)+tuple(
            getattr(result,n) for n in surface_names)

model=Transition().eval()
torch.manual_seed(20261029)
x=torch.randn(1,32)
scratch=[resident[n].clone() for n in scratch_names]
pop=[torch.zeros(1,4,32),torch.zeros(1,4),torch.zeros(1,4),torch.zeros(1,4,8)]
args=(x,torch.zeros(1,dtype=torch.long),*scratch,*pop,*[resident[n].clone() for n in state_names])
names=['encoded','echo_hash']+['scratch_'+n for n in scratch_names]+['hypotheses_'+n for n in pop_names]+['state_'+n for n in state_names]
outputs=['next_'+n for n in scratch_names]+['next_hypotheses_'+n for n in pop_names]+surface_names
out=ROOT/'core-export';out.mkdir(exist_ok=True)
with torch.no_grad():
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        torch.onnx.export(model,args,str(out/'treyrin-core-auto.onnx'),dynamo=False,
            input_names=names,output_names=outputs,opset_version=18,
            do_constant_folding=False,export_params=True)
    (out/'export-warnings.json').write_text(json.dumps([str(n.message) for n in notices],ensure_ascii=False,indent=2),encoding='utf-8')

save_file({n:v.contiguous() for n,v in zip(names,args)},str(out/'native-inputs.safetensors'))
print('Exported complete configured cognitive graph, including the enabled amygdala path. No comparisons or evaluation jobs executed.',flush=True)
