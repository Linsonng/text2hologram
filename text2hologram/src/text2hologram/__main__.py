from text2hologram.settings import load_settings, update_settings
from text2hologram.device_setup import setup_device
from text2hologram.model import load_model
from text2hologram.image_generation import generate_images
from text2hologram.post_processing import process_depth_map, cgh
from text2hologram.super_resolution import super_resolve_image
import argparse
import numpy as np
import cv2
import torch


# This function loads the default settings from the `config.json` file,
# defines and parses the command line arguments, and updates the settings
# based on the command line arguments.
def parse_arguments():
    # Load default settings from `config.json`
    settings = load_settings()
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--device', default=settings['general']['device'], help='Device')
    parser.add_argument('--iterations', type=int, default=settings['general']['iterations'], help='Number of iterations')
    parser.add_argument('--inference_steps', type=int, default=settings['diffusion']['inference_steps'], help='Number of Stable Diffusion inference steps')
    parser.add_argument('--outputdir', default=settings['general']['output directory'], help='Output directory')
    parser.add_argument('--super_reso', default=settings['diffusion']['super_reso'], help='Use super resolution')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Update settings based on the parsed command line arguments
    settings = update_settings(settings, args)
    
    return settings

# The main function. It parses the command line arguments,
# sets up the device, loads the model and necessary transforms, generates images,
# processes the depth map, performs CGH, and waits for the user to press Enter to exit.
def main():
    # Parse command line arguments and load updated settings
    settings = parse_arguments()
    
    # Set up the device
    device = setup_device(settings)
    
    # Load the model and necessary transforms
    midas, pipe, transform = load_model(device)
    
    # Get user input for the sentence to base the image creation on
    prompt = input("Enter a sentence, which the image is going to be created based on : ")
    
    # Generate images
    image_name, images = generate_images(pipe, prompt, settings, device)

    if settings['diffusion']['super_reso']:
        print("Super-resolution is applied to create 4K images, requiring more memory. If crashing, use --super_reso=false to disable.")
        image_sr = super_resolve_image(image_name)
        
        settings['general']['iterations'] = 200
        settings['image']['zero mode distance'] = 0.15
        settings['slm']['pixel pitch'] = 0.00000374
        settings['slm']['resolution'] = [2400,4094]
        settings['beam']['wavelength'] = 0.000000518



    else:

        print("Super-resolution is off. Use --super_reso=true for 4K images. For memory issues, try on Colab:https://github.com/Linsonng/text2hologram/tree/main")
        image_sr = images[0]
        settings['slm']['resolution'] =  settings['diffusion']['resolution']

    # Process the depth map
    process_depth_map(midas, transform, np.array(image_sr), device, settings)
    

    torch.cuda.empty_cache()
    # Perform CGH
    cgh(settings, image_name)
    
    # Wait for the user to press Enter to exit
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
