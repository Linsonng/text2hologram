import torch
import logging
logging.getLogger().setLevel(logging.ERROR)
from diffusers import StableDiffusionPipeline

def load_model(device):
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    model_type = "DPT_Large"  
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    scheduler = None
    model_revision = None
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        revision=model_revision,
        )
    pipe = pipe.to(device)
    transform = midas_transforms.dpt_transform
    return midas, pipe, transform
