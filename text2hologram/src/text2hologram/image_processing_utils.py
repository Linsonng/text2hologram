from pathlib import Path
import urllib.request
import cv2
import numpy as np
from PIL import Image

def download_file(url: str, path: Path) -> None:
    '''Download file from a given URL to the specified path.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)

def fetch_model_files(model_name: str, base_dir: Path):
    '''Download model files if they do not exist locally.'''
    model_xml_path = base_dir / f'{model_name}.xml'
    model_bin_path = base_dir / f'{model_name}.bin'
    
    if not model_xml_path.exists():
        base_url = f'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP16/'
        download_file(base_url + f'{model_name}.xml', model_xml_path)
        download_file(base_url + f'{model_name}.bin', model_bin_path)
    else:
        print(f'{model_name} already downloaded to {base_dir}')
    return model_xml_path

def to_rgb(image_data: np.ndarray) -> np.ndarray:
    '''Convert image_data from BGR to RGB.'''
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def convert_result_to_image(result: np.ndarray) -> np.ndarray:
    '''Convert network result to image format with values from 0-255.'''
    result = result.squeeze(0).transpose(1, 2, 0) * 255
    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)

def process_and_get_image_crops(image_crops, compiled_model, target_width, target_height, input_width, input_height):
    '''Process each image crop, visualize the results, and return the processed crops.'''
    processed_crops = []
    bicubic_crops = []
    
    for image_crop in image_crops:
        bicubic_image = cv2.resize(image_crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)
        
        image_crop_resized = cv2.resize(image_crop, (input_width, input_height))
        input_image_original = np.expand_dims(image_crop_resized.transpose(2, 0, 1), axis=0)
        
        original_image_key, bicubic_image_key = compiled_model.inputs
        output_key = compiled_model.output(0)
        
        result = compiled_model(
            {
                original_image_key.any_name: input_image_original,
                bicubic_image_key.any_name: input_image_bicubic,
            }
        )[output_key]
        
        result_image = convert_result_to_image(result)
        processed_crops.append(result_image)
        bicubic_crops.append(bicubic_image)
    return processed_crops, bicubic_crops

def combine_image_crops(crops: list) -> np.ndarray:
    '''Combine the four image crops into a full image.'''
    top_image = np.hstack((crops[0], crops[2]))
    bottom_image = np.hstack((crops[1], crops[3]))
    return np.vstack((top_image, bottom_image))
