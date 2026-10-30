"""One joint policy with actual cortical/drive input, reusing existing neurons.

Research implementation, not an acceptance claim. There is one softmax over
(goal, action); no external action switch, keyword/time rule, or new model.
The two neural logit branches are existing parameters, not two routed agents.
"""
import torch
from treyrin.goal_control import joint_softmax


def joint_context_policy(workspace,state,features,context,allowed):
    vectors=torch.cat((state.goal_vectors,state.goal_vectors.new_zeros(state.goal_vectors.shape[0],1,workspace.width)),1)
    b,rows,_=vectors.shape
    current=state.working[:,-1,None].expand(-1,rows,-1)
    drives=state.drive_state if state.drive_state is not None else current.new_zeros(b,3)
    public_clock=state.cognitive_context[:,None].expand(-1,rows,-1)
    local=torch.cat((current,vectors,drives[:,None].expand(-1,rows,-1),public_clock),-1)
    # Reuse the already-present working/drive controller as a differentiable
    # residual. New goal features and semantic vectors still enter the same
    # joint decision, including the null row's self-directed opportunities.
    residual=workspace.control(local)
    goal_logits=workspace.goal_joint.logits(features,context,vectors)
    logits=goal_logits+workspace.goal_gain.sigmoid()*residual
    probs=joint_softmax(logits,allowed,null_row=rows-1,null_action=0)
    choice=probs.flatten(1).argmax(-1);row=choice//probs.shape[-1];action=choice%probs.shape[-1]
    return dict(probabilities=probs,row=row,action=action,logits=logits,context_logits=residual,
                focused_goal=focused_goal(state,row))


def focused_goal(state,row):
    goals=state.goal_vectors;slots=goals.shape[1]
    if row.shape!=(goals.shape[0],)or bool(((row<0)|(row>slots)).any()):raise ValueError('Actual selected goal row required')
    selected=goals.gather(1,row.clamp_max(slots-1)[:,None,None].expand(-1,1,goals.shape[-1])).squeeze(1)
    return selected*(row<slots)[:,None]


def resource_legal(mask,resource):
    """An unavailable compute budget permits waiting only, irrespective of utility."""
    if not 0<=resource<=1:raise ValueError('Normalized available resource required')
    out=mask.clone()
    if resource==0:out[:,:,1:]=False
    return out
