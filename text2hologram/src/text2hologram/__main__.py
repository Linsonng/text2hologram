from text2hologram.settings import load_settings, update_settings
from text2hologram.device_setup import setup_device
from text2hologram.model import load_model
from text2hologram.image_generation import generate_images
from text2hologram.post_processing import process_depth_map, cgh
import argparse
import numpy as np

def parse_arguments():
    # Load default settings from config.json
    settings = load_settings()
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--iterations', type=int, default=settings['general']['iterations'], help='Number of iterations')
    parser.add_argument('--inference_steps', type=int, default=settings['diffusion']['inference_steps'], help='Number of Stable Diffusion inference steps')
    parser.add_argument('--outputdir', default=settings['general']['output directory'], help='Output directory')
    # Parse command line arguments
    args = parser.parse_args()
    settings = update_settings(settings, args)
    return settings

def main():
    settings = parse_arguments()
    device = setup_device(settings)
    midas, pipe, transform = load_model(device)
    prompt = input("Enter a sentence, which the image is going to be created based on : ")
    image_name, images = generate_images(pipe, prompt, settings)
    process_depth_map(midas, transform, np.array(images[0]), device, settings)
    cgh(settings, image_name)
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
