"""Equivalent Boolean Where lowering for the bundled native ONNX runtime."""
import sys,shutil
from pathlib import Path
import onnx
from onnx import helper,TensorProto
root=Path(__file__).parent/'core-export'
backup=root/'original-memory-trace';backup.mkdir(exist_ok=True)
for filename in ['treyrin-observe.onnx','treyrin-consolidate.onnx','treyrin-candidates.onnx']:
    path=root/filename
    if not (backup/filename).exists():shutil.copy2(path,backup/filename)
    model=onnx.load(backup/filename)
    typed=onnx.shape_inference.infer_shapes(model)
    types={v.name:v.type.tensor_type.elem_type for v in [*typed.graph.input,*typed.graph.value_info,*typed.graph.output]}
    types.update({v.name:v.data_type for v in typed.graph.initializer})
    nodes=[];count=0
    for node in model.graph.node:
        if node.op_type=='Where' and types.get(node.input[1])==TensorProto.BOOL:
            prefix='treyrin_bool_where_'+str(count);condition,x,y=node.input
            nodes.extend([helper.make_node('Not',[condition],[prefix+'_not']),
                helper.make_node('And',[condition,x],[prefix+'_yes']),
                helper.make_node('And',[prefix+'_not',y],[prefix+'_no']),
                helper.make_node('Or',[prefix+'_yes',prefix+'_no'],list(node.output),name=node.name)])
            count+=1
        else:nodes.append(node)
    del model.graph.node[:];model.graph.node.extend(nodes)
    onnx.save(model,path)
    print(filename,'Boolean Where operators lowered:',count,flush=True)
