import os
import numpy as np
import torch
from PIL import Image
import odak


def process_depth_map(midas, transform, img, device, settings):
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output_depth = prediction.cpu().numpy()
    output_depth = ((output_depth - output_depth.min()) / (output_depth.max()-output_depth.min())) * 255
    output_depth = output_depth.astype(np.uint8)
    im = Image.fromarray(output_depth)
    im.convert('RGB').save(settings['general']['output directory']+"/"+"depthmap.jpg")
    print("Depth acquired.") 



def cgh(settings,image_name):
    
    settings = settings
    odak.tools.check_directory(settings["general"]["output directory"])
    ch = settings["target"]["color channel"]
    target_image = odak.learn.tools.load_image(image_name, normalizeby = 255.)[: , :, ch]
    target_depth = 1. - odak.learn.tools.load_image(settings["general"]["output directory"]+"/"+settings["target"]["depth filename"], normalizeby = 255.)
    if len(target_depth.shape) > 2:
        target_depth = target_depth[:, :, 1]
    device = torch.device(settings["general"]["device"])
    loss_function = odak.learn.wave.multiplane_loss(
                                                    target_image = target_image.unsqueeze(0),
                                                    target_depth = target_depth,
                                                    target_blur_size = settings["target"]["defocus blur size"],
                                                    number_of_planes = settings["target"]["number of planes"],
                                                    multiplier = settings["target"]["multiplier"],
                                                    blur_ratio = settings["target"]["blur ratio"],
                                                    weights = settings["target"]["weights"],
                                                    scheme = settings["target"]["scheme"],
                                                    device = device
                                                   )
    targets, focus_target, depth = loss_function.get_targets()
    for hologram_id in range(settings["general"]["hologram number"]):
        optimizer = odak.learn.wave.multiplane_hologram_optimizer(
                                                                  wavelength = settings["beam"]["wavelength"],
                                                                  image_location = settings["image"]["location"],
                                                                  image_spacing = settings["image"]["delta"],
                                                                  slm_pixel_pitch = settings["slm"]["pixel pitch"],
                                                                  slm_resolution = settings["slm"]["resolution"],
                                                                  targets = targets,
                                                                  propagation_type = settings["general"]["propagation type"],
                                                                  number_of_iterations = settings["general"]["iterations"],
                                                                  learning_rate = settings["general"]["learning rate"],
                                                                  number_of_planes = settings["target"]["number of planes"],
                                                                  mask_limits = settings["target"]["mask limits"],
                                                                  zero_mode_distance = settings["image"]["zero mode distance"],
                                                                  loss_function = loss_function,
                                                                  device = device
                                                                 )
        phase, _, reconstructions = optimizer.optimize()
        save(settings, device, phase, reconstructions, targets, focus_target.squeeze(0), depth, hologram_id)

def save(settings, device, phase, reconstructions, targets, focus_target, depth, hologram_id):
    torch.no_grad()
    for plane_id in range(settings["target"]["number of planes"]):
        odak.learn.tools.save_image(
                                    settings["general"]["output directory"] + "/target_{:04}.png".format(plane_id),
                                    targets[plane_id], cmin = 0, cmax = 1.)
    odak.learn.tools.save_image(
                                settings["general"]["output directory"] + "/target.png",
                                focus_target, cmin = 0., cmax= 1.)
    odak.learn.tools.save_image(
                                settings["general"]["output directory"] + '/depth.png',
                                depth,
                                cmin = 0.,
                                cmax = 1.
                               )
    checker_complex = odak.learn.wave.linear_grating(
                                                     settings["slm"]["resolution"][0],
                                                     settings["slm"]["resolution"][1],
                                                     add = odak.pi,
                                                     axis='y'
                                                    ).to(device)
    checker = odak.learn.wave.calculate_phase(checker_complex)
    phase_grating = phase + checker
    phase_normalized = ((phase % (2 * odak.pi)) / (2 * odak.pi)) * 255
    odak.learn.tools.save_image(settings["general"]["output directory"]+"/phase_{:04}.png".format(hologram_id), phase_normalized)
    phase_grating = ((phase_grating % (2 * odak.pi)) / (2 * odak.pi)) * 255
    odak.learn.tools.save_image(settings["general"]["output directory"]+"/phase_grating_{:04}.png".format(hologram_id), phase_grating)
    if hologram_id == 0:
        k = settings["target"]["number of planes"]
        t = range(k)
        for plane_id in t:
            odak.learn.tools.save_image(
                                        settings["general"]["output directory"]+"/recon_{:04}.png".format(plane_id),
                                        reconstructions[plane_id], cmin = 0, cmax = 1.)
    data = {
            "targets" : focus_target,
            "depth" : depth,
            "phases" : phase_normalized / 255.,
            "settings" : settings
           }
    odak.learn.tools.save_torch_tensor('{}/data.pt'.format(settings["general"]["output directory"]), data)
    print('Outputs stored at ' + os.path.expanduser(settings["general"]["output directory"]))

def perform_cgh_and_rename(settings, image_path, output_directory, new_file_name, wavelength=None, color_channel=None):
    if wavelength:
        settings['beam']['wavelength'] = wavelength
    if color_channel:
        settings['target']['color channel'] = color_channel
    
    torch.cuda.empty_cache()
    cgh(settings, image_path)
    
    default_output_file = os.path.join(output_directory, 'phase_0000.png')
    renamed_output_file = os.path.join(output_directory, new_file_name)
    
    if os.path.exists(default_output_file):
        os.rename(default_output_file, renamed_output_file)