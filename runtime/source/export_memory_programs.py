"""Compile the existing observation/consolidation functions. No evaluation."""
from pathlib import Path
import sys,json,warnings
root=Path(__file__).parent
source=(root/'export_model_core.py').read_text('utf-8')
exec(compile(source[:source.index('model=Transition().eval()')],str(root/'export_model_core.py'),'exec'))
from torch.onnx import symbolic_helper
def native_argsort(g,value,*args):
    # Occupied hot episodes have distinct monotonic event IDs. Ties are empty
    # slots, masked out by the original consolidation function.
    dim,descending=args[-2:]
    axis=symbolic_helper._parse_arg(dim,'i')
    largest=int(symbolic_helper._parse_arg(descending,'b'))
    shape=g.op('Shape',value)
    k=g.op('Gather',shape,g.op('Constant',value_t=torch.tensor([axis],dtype=torch.long)),axis_i=0)
    _,indices=g.op('TopK',value,k,axis_i=axis,largest_i=largest,sorted_i=1,outputs=2)
    return indices
torch.onnx.register_custom_op_symbolic('aten::argsort',native_argsort,18)

class Observe(torch.nn.Module):
    def __init__(self):super().__init__();self.core=core
    def forward(self,x,source,echo_hash,*values):
        state=ExperienceState(**dict(zip(state_names,values)),binding='native-memory')
        pop=HypothesisPopulation(x.new_zeros(1,4,32),x.new_zeros(1,4),x.new_zeros(1,4),x.new_zeros(1,4,8))
        step=self.core.step(x,state.working,state,None,selected_goal_context(state),
            regional_state=(state.cortex,state.relay,state.selection),
            hippocampal_state=state.hippocampal_population,
            cerebellar_prediction_prev=state.cerebellar_prediction,da_prev=state.dopamine,
            hypothesis_state=pop,echo_query_hash=echo_hash,drift=x.new_zeros(1),amygdala_mode='normal')
        result=self.core.commit_observation(x,source,step.working,state,
            regions=(step.cortex,step.relay,step.selection),hippocampal_population=step.hippocampal_population,
            thalamic=step.thalamic,cerebellar_prediction=step.cerebellar_prediction,
            dopamine=step.dopamine,neuromod=step.neuromod,mismatch=step.hippocampal_mismatch,
            echo_hash_in=echo_hash)
        return tuple(result.tensors()[n] for n in state_names)

class Consolidate(torch.nn.Module):
    def __init__(self):super().__init__();self.core=core
    def forward(self,*values):
        state=ExperienceState(**dict(zip(state_names,values)),binding='native-memory')
        result=self.core.consolidate(state)
        return tuple(result.tensors()[n] for n in state_names)

out=root/'core-export'
inputs=['encoded','source','echo_hash']+['state_'+n for n in state_names]
outputs=['state_'+n for n in state_names]
args=(torch.zeros(1,32),torch.zeros(1,32),torch.zeros(1,dtype=torch.long),*[resident[n].clone() for n in state_names])
with torch.no_grad(),warnings.catch_warnings(record=True) as messages:
    warnings.simplefilter('always')
    torch.onnx.export(Observe().eval(),args,str(out/'treyrin-observe.onnx'),dynamo=False,
        input_names=inputs,output_names=outputs,opset_version=18,do_constant_folding=False)
    torch.onnx.export(Consolidate().eval(),args[3:],str(out/'treyrin-consolidate.onnx'),dynamo=False,
        input_names=inputs[3:],output_names=outputs,opset_version=18,do_constant_folding=False)
(out/'memory-export-warnings.json').write_text(json.dumps([str(m.message) for m in messages],ensure_ascii=False,indent=2),encoding='utf-8')
save_file({n:v.contiguous() for n,v in zip(inputs,args)},str(out/'memory-inputs.safetensors'))
print('Compiled original neural observation and consolidation functions; no model answers or comparisons run.',flush=True)
