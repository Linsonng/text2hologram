# Text2Hologram

## Overview
The Text2Hologram project aims to generate holographic images based on textual input. It uses various machine learning models and image processing techniques to achieve this.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [__main__.py](#__main__py)
- [color_recon.py](#color_reconpy)
- [config.json](#configjson)
- [device_setup.py](#device_setuppy)
- [image_generation.py](#image_generationpy)
- [image_processing_utils.py](#image_processing_utilspy)
- [model.py](#modelpy)
- [post_processing.py](#post_processingpy)
- [settings.py](#settingspy)
- [super_resolution.py](#super_resolutionpy)
- [utils.py](#utilspy)
- [Contributing](#contributing)
- [License](#license)


## Usage
After installing the project, you can run the main script as follows:

```bash
python __main__.py [options]
```

## Options
- `--device`: The computing device to use (CPU or GPU).
- `--iterations`: Number of iterations for image generation.
- `--inference_steps`: Number of Stable Diffusion inference steps.
- `--outputdir`: The directory to save the generated images.
- `--super_reso`: Enable or disable super-resolution.

## Main Functions

### __main__.py
- `main()`: The main function that ties together all the other modules.
- `parse_arguments()`: Parses command line arguments and updates settings accordingly.


```
python __main__.py [options]
```

### color_recon.py
- `color_reconstruction()`: Function responsible for the color reconstruction process. The input image is a pure phase hologram.


```python
from color_recon import color_reconstruction
color_reconstruction(input_image)
```

### device_setup.py
- `setup_device(settings)`: Configures the computing device based on the settings.

### image_generation.py
- `generate_images(pipe, prompt, settings, device)`: Generates images based on the textual input. 'pipe' is a generative model pipeline that allows switching to other pipelines. 'prompt' is the text entered by the user.


```python
from image_generation import generate_images
generate_images(pipe, prompt, settings, device)
```

### image_processing_utils.py
- `process_and_get_image_crops(image, num_crops)`: Processes an image and returns a specified number of cropped images. This function is essential for image preprocessing before feeding it into the machine learning models.

This module also contains other utility functions for image processing that serve various purposes throughout the application.


### model.py
- `load_model(device)`: Loads the machine learning model and necessary transforms.


```python
from model import load_model
model, pipe, transform = load_model(device)
```

### post_processing.py
- `process_depth_map()`: Processes the depth map for the generated image.

```python
from post_processing import process_depth_map
process_depth_map(midas, transform, image_sr, device, settings)

```

- `cgh()`: Performs Computer Generated Holography on the image.


```python
from post_processing import cgh
cgh(settings, image_name)
```

### settings.py
- `load_settings()`: Loads the settings from a JSON file.
- `update_settings()`: Updates the settings based on user input.


### super_resolution.py
- `super_resolve_image(image_name)`: Applies super-resolution to the image.


```python
from super_resolution import super_resolve_image
super_resolve_image(image_name)
```

### utils.py
- `display_images_from_directory()`: Displays images from a directory.
- `create_dirs()`: Creates directories if they don't exist.
- `combine_rgb_images()`: Combines RGB channels into a single image.

### Contributing
If you want to contribute, feel free to open an issue or submit a pull request.

