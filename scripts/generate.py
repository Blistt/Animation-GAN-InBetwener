"""
This script generates an image using a chosen model and a checkpoint.
"""

import torch
import pathlib
from PIL import Image
from torchvision import transforms
from generators.generator_padded import UNetPadded
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from utils.utils import create_gif



def generate(model, checkpoint_path, input_triplet_path, binary_thresh=0.0, crop_shape=None, 
             save_path=None, filename='in_between'):
    transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize((512, 512), antialias=True),])

    frame1 = transform(Image.open(pathlib.Path(input_triplet_path) / 'frame1.png'))
    frame2 = transform(Image.open(pathlib.Path(input_triplet_path) / 'frame2.png'))
    frame3 = transform(Image.open(pathlib.Path(input_triplet_path) / 'frame3.png'))

    # Add batch dimension
    frame1 = frame1.unsqueeze(0)
    frame2 = frame2.unsqueeze(0)
    frame3 = frame3.unsqueeze(0)
    
    # Binarizes frames
    frame1 = (frame1 > 0.75).float()
    frame2 = (frame2 > 0.75).float()
    frame3 = (frame3 > 0.75).float()

    # Loads a model from a checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Generates in-between frame for a given triplet
    in_between = model(frame1, frame3)

    # performs center crop if specified with torch center crop
    if crop_shape:
        in_between = transforms.CenterCrop(crop_shape)(in_between)
        frame1 = transforms.CenterCrop(crop_shape)(frame1)
        frame2 = transforms.CenterCrop(crop_shape)(frame2)
        frame3 = transforms.CenterCrop(crop_shape)(frame3)

    if binary_thresh > 0.0:
        in_between[in_between >= binary_thresh] = 1.0
        in_between[in_between < binary_thresh] = 0.0

    if save_path:
        save_image(in_between, pathlib.Path(save_path) / (filename + '.png'))
        create_gif(frame1, frame2, frame3, in_between, save_path + '/' + filename, 0)

    return in_between