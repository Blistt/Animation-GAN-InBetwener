"""
This script generates an image using a chosen model and a checkpoint.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd())) if str(Path.cwd()) not in sys.path else None
import torch
from torchvision import transforms
from generators.generator_padded import UNetPadded
from line_extractor.model import SketchKeras
from line_extractor.extract import extract_from_video
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def generate(model, frame1, frame2, binary_thresh=0.0, crop_shape=None, device='cuda:0'):

    transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1),
                         ])

    frame1 = transform(frame1).to(device)
    frame3 = transform(frame2).to(device)  

    # Add batch dimension
    frame1 = frame1.unsqueeze(0)
    frame3 = frame3.unsqueeze(0)
    
    # Binarizes frames
    frame1 = (frame1 > 0.75).float()
    frame3 = (frame3 > 0.75).float()


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

    # if tensor in cuda, move it to cpu
    if in_between.is_cuda:
        in_between = in_between.cpu()
    in_between = in_between.squeeze(0).squeeze(0).detach().numpy()

    return in_between