from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader

from transformers import PretrainedConfig, WhisperConfig

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

import json
from pathlib import Path
from tensorrt_llm.logger import logger
from typing import Any, Optional


###################
# Mistral code here
# https://github.com/vllm-project/vllm/blob/e1098ced95146d98a4ed46c81ee709013d54fb1f/vllm/transformers_utils/configs/mistral.py
###################

def adapt_config_dict(config_dict: dict[str, Any], **kwargs) -> PretrainedConfig:
    config_dict.update(kwargs)
    config_dict = _remap_general_mistral_args(config_dict)

    if bool(config_dict.get("quantization")):
        config_dict = _remap_mistral_quantization_args(config_dict)

    if bool(config_dict.get("moe")):
        config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]

    if bool(config_dict.get("yarn")):
        config_dict = _remap_mistral_yarn_args(config_dict)

    is_vision = (config_dict.get("multimodal") or {}).get(
        "vision_encoder_args"
    ) or config_dict.get("vision_encoder")
    is_audio = bool(
        ((config_dict.get("multimodal") or {}).get("whisper_model_args") or {}).get(
            "encoder_args"
        )
    )

    assert not (is_vision and is_audio), "Vision and audio are mutually exclusive"

    if is_vision:
        config_dict = _remap_mistral_vision_args(config_dict)
    if is_audio:
        config_dict = _remap_mistral_audio_args(config_dict)

    config = PretrainedConfig.from_dict(config_dict)

    logger.debug("Initialized config %s", config)

    return config


def _remap_mistral_vision_args(config: dict) -> dict:
    if config.get("multimodal"):
        vision_config = config.pop("multimodal")
    else:
        vision_config = config.pop("vision_encoder")

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "pixtral",
        "architectures": ["PixtralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "vision_config": PretrainedConfig.from_dict(vision_config),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_mistral_yarn_args(config: dict) -> dict:
    # Direct remaps: yarn.X -> rope_scaling.Y
    # Source keys are from mistral.model.args.YarnArgs
    _map = {
        "beta": "beta_fast",
        "alpha": "beta_slow",
    }
    yarn_config = config.get("yarn") or {}
    renamed_yarn_config = {_map.get(k, k): v for k, v in yarn_config.items()}
    config["rope_scaling"] = {
        "rope_type": "yarn",
        "mscale_all_dim": 1,  # We hardcoded this to 1
        **renamed_yarn_config,
    }
    return config


def _remap_general_mistral_args(config: dict) -> dict:
    # Mistral key -> HF key
    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }
    # HF key -> (Mistral key, default value)
    top_level_mapping_with_default = {
        "model_type": ("model_type", "transformer"),
        "hidden_act": ("activation", "silu"),
        "tie_word_embeddings": ("tied_embeddings", False),
        "max_seq_len": ("max_seq_len", 128_000),
        "max_position_embeddings": ("max_position_embeddings", 128_000),
    }

    for key, new_key in config_mapping.items():
        if key in config:
            config[new_key] = config.pop(key)

    for new_key, (key, default_value) in top_level_mapping_with_default.items():
        config[new_key] = config.pop(key, default_value)

    return config


def _remap_mistral_quantization_args(config: dict) -> dict:
    quantization = config.get("quantization", {})
    if quantization.get("qformat_weight") == "fp8_e4m3":
        # This maps to the FP8 static per-tensor quantization scheme
        quantization_config = {"quant_method": "fp8", "activation_scheme": "static"}
    elif quantization.get("quant_method") == "compressed-tensors":
        # Pass through the quantization config to compressed-tensors
        quantization_config = quantization
    else:
        raise ValueError(f"Found unknown quantization='{quantization}' in config")

    config["quantization_config"] = quantization_config

    return config


def _remap_mistral_audio_args(config: dict) -> dict:
    whisper_args = config["multimodal"].pop("whisper_model_args")
    encoder_args = whisper_args["encoder_args"]
    downsample_args = whisper_args["downsample_args"]

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "whixtral",
        "architectures": ["VoxtralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "audio_config": WhisperConfig(
            num_mel_bins=encoder_args["audio_encoding_args"]["num_mel_bins"],
            window_size=encoder_args["audio_encoding_args"]["window_size"],
            sampling_rate=encoder_args["audio_encoding_args"]["sampling_rate"],
            hop_length=encoder_args["audio_encoding_args"]["hop_length"],
            downsample_factor=downsample_args["downsample_factor"],
            d_model=encoder_args["dim"],
            encoder_layers=encoder_args["n_layers"],
            encoder_ffn_dim=encoder_args["hidden_dim"],
            encoder_attention_heads=encoder_args["n_heads"],
            vocab_size=encoder_args["vocab_size"],
            max_source_positions=encoder_args["max_source_positions"],
            is_encoder_decoder=False,  # Override WhisperConfig default
        ),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config

######################
# End of Mistral code
######################


@register_config_loader("mistral")
class MistralConfigLoader(BaseConfigLoader):
    def _load_mistral_config_dict(
            self, checkpoint_dir: str, config_file_name: str) -> Optional[dict]:
        file_path = Path(checkpoint_dir) / Path(config_file_name)

        if file_path.exists() and file_path.is_file():
            with open(file_path) as file:
                return json.load(file)
        return None
    
    # Adaptation of
    # https://github.com/vllm-project/vllm/blob/e1098ced95146d98a4ed46c81ee709013d54fb1f/vllm/transformers_utils/config.py#L171    
    def _parse_mistral_config(self, checkpoint_dir: str):
        config_file_name = "params.json"

        config_dict = self._load_mistral_config_dict(checkpoint_dir, config_file_name)
        if config_dict is None:
            raise ValueError(
                f"Failed to load '{config_file_name}' config from '{checkpoint_dir}'. "
                f"Only local checkpoints are supported for mistral format."
            )
        assert isinstance(config_dict, dict)

        # https://github.com/vllm-project/vllm/blob/e1098ced95146d98a4ed46c81ee709013d54fb1f/vllm/transformers_utils/config.py#L1090
        if (
            max_position_embeddings := config_dict.get("max_position_embeddings")
        ) is None:
            max_position_embeddings = 128_000
            config_dict["max_position_embeddings"] = max_position_embeddings
        
        pretrained_config = adapt_config_dict(config_dict)
        
        # Mistral configs may define sliding_window as list[int]. Convert it
        # to int and add the layer_types list[str] to make it HF compatible
        if (sliding_window := getattr(pretrained_config, "sliding_window", None)) and isinstance(
            sliding_window, list
        ):
            pattern_repeats = pretrained_config.num_hidden_layers // len(sliding_window)
            layer_types = sliding_window * pattern_repeats
            pretrained_config.layer_types = [
                "full_attention" if layer_type is None else "sliding_attention"
                for layer_type in layer_types
            ]
            pretrained_config.sliding_window = next(filter(None, sliding_window), None)
            
        return config_dict, pretrained_config
    

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:

        config_dict, pretrained_config = self._parse_mistral_config(checkpoint_dir)

        moe_backend = kwargs.get('moe_backend', 'CUTLASS')

        quant_config = QuantConfig()
        layer_quant_config = None
        if hasattr(pretrained_config, "quantization_config"):
            hf_quant_config = pretrained_config.quantization_config
            quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
                hf_quant_config, moe_backend)
            
        return ModelConfig(pretrained_config=pretrained_config, 
                           quant_config=quant_config, 
                           quant_config_dict=layer_quant_config)