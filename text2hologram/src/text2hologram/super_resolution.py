from text2hologram.image_processing_utils import *
from openvino.runtime import Core

def super_resolve_image(image_name: str) -> None:
    '''Process the provided image using super resolution, and save the result.'''
    # Define constants
    model_name = 'single-image-super-resolution-1032'
    base_model_dir = Path("./model").expanduser()

    # Fetch model files
    model_xml_path = fetch_model_files(model_name, base_model_dir)


    ie = Core()
    model = ie.read_model(model=model_xml_path)
    compiled_model = ie.compile_model(model=model, device_name='AUTO')

    # Extract network input and output details
    original_image_key, bicubic_image_key = compiled_model.inputs
    output_key = compiled_model.output(0)
    input_height, input_width = list(original_image_key.shape)[2:]
    target_height, target_width = list(bicubic_image_key.shape)[2:]
    upsample_factor = int(target_height / input_height)

    # Process image
    full_image = cv2.imread(image_name)
    full_w, full_h = full_image.shape[:2]

    # Split the image into 4 crops
    image_crops = [
        full_image[0:full_w//2, 0:full_h//2, :],
        full_image[full_w//2:, 0:full_h//2, :],
        full_image[0:full_w//2, full_h//2:, :],
        full_image[full_w//2:, full_h//2:, :]
    ]

    # Using the updated function to get processed crops
    processed_crops, bicubic_crops = process_and_get_image_crops(image_crops, compiled_model, target_width, target_height, input_width, input_height)
    full_combined_processed_image = combine_image_crops(processed_crops)

    # Convert the processed image from BGR to RGB and save it
    img_pil = Image.fromarray(cv2.cvtColor(full_combined_processed_image, cv2.COLOR_BGR2RGB))
    img_pil = img_pil.resize((4094,2400), Image.BICUBIC)
    img_pil.save(image_name)

    return img_pil

