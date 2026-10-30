"""Optional Transformers registry entry point for the Treyrin model."""
from pathlib import Path
__path__=[str(Path(__file__).resolve().parent)]
__package__=__name__
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from .configuration_treyrin import TreyrinConfig
from .modeling_treyrin import TreyrinForCausalLM
__version__='5.0.0'
AutoConfig.register('treyrin',TreyrinConfig,exist_ok=True)
AutoModel.register(TreyrinConfig,TreyrinForCausalLM,exist_ok=True)
AutoModelForCausalLM.register(TreyrinConfig,TreyrinForCausalLM,exist_ok=True)
