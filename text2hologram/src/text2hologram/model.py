import torch
import logging
# Suppress logging messages for a cleaner output
logging.getLogger().setLevel(logging.ERROR)
from diffusers import StableDiffusionPipeline

# Function to load the model and necessary transforms.
# The function takes device (either "cuda" or "cpu") as an argument,
# loads the MiDaS model, applies necessary transforms,
# and loads the StableDiffusionPipeline model with the appropriate dtype
def load_model(device,model_id = "runwayml/stable-diffusion-v1-5",print_id = False):
    # Define the model_id and model_type for loading the MiDaS model

    model_type = "DPT_Large"  
    
    if print_id:
        print(model_id,model_type)
    
    # Load the MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    # Move the model to the specified device
    midas.to(device)
    
    # Load the MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    model_revision = None
    
    # Get the appropriate dtype for tensor computations based on the device
    dtype = setup_dtype(device)
    
    # Load the StableDiffusionPipeline model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        revision=model_revision,
        )
    
    # Move the pipeline model to the specified device
    pipe = pipe.to(device)
    
    # Load the transform for the MiDaS model
    transform = midas_transforms.dpt_transform
    
    # Return the MiDaS model, pipeline model, and the transform
    return midas, pipe, transform

# This function determines the data type (dtype) for tensor computations
# based on the specified device. 
# If the device is a CUDA-enabled GPU, it returns `float16`, which can 
# leverage Tensor Cores for faster computations on such devices.
# If the device is a CPU, it returns `float32`, as `float16` operations 
# are not natively supported on CPUs in PyTorch.
def setup_dtype(device):
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dtype
