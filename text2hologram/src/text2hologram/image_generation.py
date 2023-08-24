import torch
import random
from text2hologram.utils import create_dirs
from PIL import Image

def generate_images(pipe, prompt, settings, device):
    seed = random.randint(0, 2147483647)
    negative_prompt = "nude, naked"
    print("Creating images.")
    images = pipe(
        prompt,
        height = settings['diffusion']['resolution'][0],
        width = settings['diffusion']['resolution'][1],
        num_inference_steps = settings['diffusion']['inference_steps'],
        guidance_scale = 9,
        num_images_per_prompt = 1,
        negative_prompt = negative_prompt,
        generator = torch.Generator(device).manual_seed(seed)
        ).images
    create_dirs([settings['general']['output directory']])
    image_name = settings['general']['output directory'] + "/"+ prompt[0:20] + ".jpg"
    images[0].save(image_name)
    print("Image saved as "+ image_name)
    return image_name, images

