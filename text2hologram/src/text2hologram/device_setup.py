import torch

def setup_device(settings):
    device = torch.device(settings['general']['device'])
    if settings['general']['device'] == "cuda" and torch.cuda.is_available():
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        raise RuntimeError("Sorry, CUDA is not available.")
    return device
