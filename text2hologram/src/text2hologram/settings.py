import json
import os

def load_settings(config_file=None):
    if config_file is None:
        # Get the directory of the current script
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Join this directory with the name of your config file
        config_file = os.path.join(dir_path, 'config.json')
    
    with open(config_file, 'r') as f:
        settings = json.load(f)
    return settings

def update_settings(settings, args):
    settings['general']['iterations'] = args.iterations
    settings['diffusion']['inference_steps'] = args.inference_steps
    settings['general']['output directory'] = args.outputdir
    return settings
