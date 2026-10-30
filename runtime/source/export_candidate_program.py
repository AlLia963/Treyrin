"""Compile the existing core with public candidate descriptions; no evaluation."""
from pathlib import Path
import warnings,json
root=Path(__file__).parent
source=(root/'export_model_core.py').read_text('utf-8')
exec(compile(source[:source.index('model=Transition().eval()')],str(root/'export_model_core.py'),'exec'))

class CandidateTransition(torch.nn.Module):
    def __init__(self):super().__init__();self.core=core
    def forward(self,x,echo_hash,candidates,choice_prior,*data):
        scratch=data[:7];pop=data[7:11];values=data[11:]
        state=ExperienceState(**dict(zip(state_names,values)),binding='native-candidate')
        result=self.core.step(x,scratch[0],state,candidates,selected_goal_context(state),
            regional_state=tuple(scratch[1:4]),hippocampal_state=scratch[4],
            cerebellar_prediction_prev=scratch[5],da_prev=scratch[6],
            echo_query_hash=echo_hash,drift=torch.zeros_like(x[:,0]),amygdala_mode='normal',
            body_bridge_context=x,body_choice_prior=choice_prior)
        return tuple(getattr(result,n) for n in scratch_names)+tuple(pop)+tuple(getattr(result,n) for n in surface_names)

out=root/'core-export'
base=load_file(str(out/'native-inputs.safetensors'))
names=['encoded','echo_hash','candidates','choice_prior']+['scratch_'+n for n in scratch_names]+['hypotheses_'+n for n in pop_names]+['state_'+n for n in state_names]
base['candidates']=torch.zeros(1,3,32);base['choice_prior']=torch.zeros(1,3)
outputs=['next_'+n for n in scratch_names]+['next_hypotheses_'+n for n in pop_names]+surface_names
axes={'candidates':{1:'candidate_count'},'choice_prior':{1:'candidate_count'}}
for n in ['native_logits','predicted_outcomes','candidate_reads','predicate_logits','action_values']:
    axes[n]={1:'candidate_count'}
with torch.no_grad(),warnings.catch_warnings(record=True) as notices:
    warnings.simplefilter('always')
    torch.onnx.export(CandidateTransition().eval(),tuple(base[n] for n in names),str(out/'treyrin-candidates.onnx'),
        dynamo=False,input_names=names,output_names=outputs,opset_version=18,do_constant_folding=False,dynamic_axes=axes)
save_file({n:base[n].contiguous() for n in names},str(out/'candidate-inputs.safetensors'))
(out/'candidate-export-warnings.json').write_text(json.dumps([str(n.message) for n in notices],indent=2),encoding='utf-8')
print('Compiled original variable-candidate transition; no answers or comparisons run.',flush=True)
