"""Import-safe adapter for the same frozen candidate and its causal roles."""
from pathlib import Path
import sys

def factory(root,checkpoint):
    root=Path(root).resolve()
    if root!=Path(__file__).resolve().parent:raise ValueError('Adapter and model sources differ')
    sys.path.insert(0,str(root))
    from role_conversation import factory as build
    return build(root,Path(checkpoint),system_prompt=None)
