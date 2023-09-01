from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images_from_directory(directory_path, ext='.jpg', size=(10, 10)):
    """
    Display images of a specific extension from the provided directory.

    Args:
    - directory_path (str): Path to the directory containing the images.
    - ext (str, optional): Extension of the images to display. Defaults to '.jpg'.
    - size (tuple, optional): Size of the displayed image. Defaults to (10, 10).
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter and sort the image files based on the provided extension
    image_files = sorted([f for f in files if f.endswith(ext)])

    # Display each image
    for image_file in image_files:
        img_path = os.path.join(directory_path, image_file)
        img = mpimg.imread(img_path)
        plt.figure(figsize=size)

        # Check if the image is grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        plt.axis('off')  # Hide axes
        plt.title(image_file)
        plt.show()

def create_dirs(paths):
    for path in paths:
        if not os.path.isdir(path) and path != '':
            os.makedirs(path)


def combine_rgb_images(img1_path, img2_path, img3_path, output_path='combined_rgb.png'):
    # Check if files exist
    if not (os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(img3_path)):
        return "One or more input file paths are invalid."

    # Load images using PIL
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img3 = Image.open(img3_path)

    # Convert the images to numpy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    img3_array = np.array(img3)

    # Check if the images are single channel (like grayscale)
    if len(img1_array.shape) == 2 and len(img2_array.shape) == 2 and len(img3_array.shape) == 2:
        # Stack the images to form a multi-channel (RGB) image
        combined = np.stack((img1_array, img2_array, img3_array), axis=-1)

        # Convert the combined numpy array back to an image and save it
        result_image = Image.fromarray(combined.astype('uint8'))
        result_image.save(output_path)

        return f"RGB image saved as {output_path}"
    else:
        return "One or more images are not single channel!"

# Example usage:
# result = combine_rgb_images('2039r.png', '2039g.png', '2039b.png', '2039rgb.png')
# print(result)

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images_from_directory(directory_path, image_names=None, title=None, ext='.jpg', size=(10, 10)):
    """
    Display images of a specific extension from the provided directory.

    Args:
    - directory_path (str): Path to the directory containing the images.
    - image_names (list, optional): List of names of specific images to display. Defaults to None.
    - title (str, optional): Title to display above the image(s). Defaults to None.
    - ext (str, optional): Extension of the images to display. Defaults to '.jpg'.
    - size (tuple, optional): Size of the displayed image. Defaults to (10, 10).
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter and sort the image files based on the provided extension
    image_files = sorted([f for f in files if f.endswith(ext)])

    # Display each image
    for image_file in image_files:
        # If specific image names are provided, skip other images
        if image_names and image_file not in image_names:
            continue

        img_path = os.path.join(directory_path, image_file)
        img = mpimg.imread(img_path)
        plt.figure(figsize=size)

        # Check if the image is grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        plt.axis('off')  # Hide axes

        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(image_file)

        plt.show()


def generate_image_names(base_name, num_images):
    """
    Generate a list of image names based on a base name and a number of images.

    Args:
    - base_name (str): The base name for the images.
    - num_images (int): The number of image names to generate.

    Returns:
    - list: A list of generated image names.
    """
    image_names = []
    for i in range(num_images):
        image_name = f"{base_name}_{str(i).zfill(4)}.png"
        image_names.append(image_name)
    return image_names


def display_selected_images(display_flag, output_directory, image_names, size=(10, 10)):
    if display_flag:
        display_images_from_directory(output_directory, image_names=image_names, title=None, ext='.png', size=size)

