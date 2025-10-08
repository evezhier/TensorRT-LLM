from typing import Optional, Any

from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import \
    HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
    MistralConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader

@register_checkpoint_loader("mistral")
class MistralCheckpointLoader(HfCheckpointLoader):
    def __init__(self,
                *,
                weight_loader: Optional[BaseWeightLoader] = None,
                weight_mapper: Optional[BaseWeightMapper] = None,
                config_loader: Optional[BaseConfigLoader] = None):
        super().__init__(weight_loader=weight_loader, 
                         weight_mapper=weight_mapper, 
                         config_loader=config_loader)
        self._checkpoint_format = "mistral"

    def load_weights(self, checkpoint_dir: str, **kwargs):
        weights = super().weight_loader.load_weights(checkpoint_dir, **kwargs)
        return self.weight_mapper.preprocess_weights(weights)
    
    def get_default_config_loader(self) -> MistralConfigLoader:
        return MistralConfigLoader()


    