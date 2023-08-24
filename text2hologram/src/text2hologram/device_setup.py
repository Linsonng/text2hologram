import torch

def setup_device(settings):
    if settings['general']['device'] == "cuda":
        if torch.cuda.is_available():
            print("CUDA is available, using CUDA.")
            device = torch.device('cuda')
        else:
            print("CUDA is selected in settings but not available, using CPU instead.")
            device = torch.device('cpu')
    else:
        print("CPU is selected in settings, using CPU.")
        device = torch.device('cpu')
    return device
