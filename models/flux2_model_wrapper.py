import torch
import numpy as np
import torch.nn.functional as F

from transformers import (
    Qwen2TokenizerFast,
    Qwen3ForCausalLM,
    AutoProcessor,
    Mistral3ForConditionalGeneration,
)
from diffusers import (
    AutoencoderKLFlux2, 
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.loaders import Flux2LoraLoaderMixin
from typing import Any, Callable, Dict, List, Optional, Tuple, Union



class Flux2ModelWrapper(Flux2LoraLoaderMixin):
    def __init__(
        self,
        ckpt_path: str = None,
        model_type: str = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.ckpt_path = ckpt_path or "black-forest-labs/FLUX.2-klein-4B"
        self.model_type = model_type or "klein"
        self.torch_dtype = torch_dtype
        self.device = torch.device(device)
        
        self.load_hf_checkpoint()
        self.to(device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") else 8
        self.default_sample_size = 128
        self.is_distilled = getattr(self, "is_distilled", False)

        print(f"✅ FluxModelWrapper loaded: {ckpt_path} ({'Klein' if self.model_type == 'klein' else 'Dev'})")

    
    def load_hf_checkpoint(self):
        kwargs = {"torch_dtype": self.torch_dtype, "variant": "fp16" if self.torch_dtype == torch.float16 else None}

        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.ckpt_path, subfolder="vae", **kwargs
        )

        self.transformer = Flux2Transformer2DModel.from_pretrained(
            self.ckpt_path, subfolder="transformer", **kwargs
        )

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.ckpt_path, subfolder="scheduler"
        )

        if self.model_type == "klein":
            # === FLUX.2 KLEIN ===
            self.text_encoder = Qwen3ForCausalLM.from_pretrained(
                self.ckpt_path, subfolder="text_encoder", **kwargs
            )
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                self.ckpt_path, subfolder="tokenizer"
            )
            self.is_distilled = True
        else:
            # === Обычный FLUX.2-dev / schnel ===
            self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                self.ckpt_path, subfolder="text_encoder", **kwargs
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                self.ckpt_path, subfolder="tokenizer"
            )


    def to(self, device: str):
        self.vae.to(device)
        self.transformer.to(device)
        self.text_encoder.to(device)
        self.device = torch.device(device)