"""
This script generates an image using a chosen model and a checkpoint.
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path.cwd())) if str(pathlib.Path.cwd()) not in sys.path else None
import torch
from torchvision import transforms
from generators.generator_padded import UNetPadded
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from utils.utils import create_gif
from PIL import Image


def generate(model, checkpoint_path, input_triplet_path, binary_thresh=0.0, crop_shape=None, 
             save_path=None, filename='in_between', gt=False):
    transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize((512, 512), antialias=True),])
    
    frame1 = transform(Image.open(pathlib.Path(input_triplet_path) / 'frame1.png'))
    frame3 = transform(Image.open(pathlib.Path(input_triplet_path) / 'frame3.png'))
    if gt:
        frame2 = transform(Image.open(pathlib.Path(input_triplet_path) / 'frame2.png'))
    else:
        frame2 = frame3.clone()

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
        create_gif(frame1, frame2, frame3, in_between, save_path + '/' + filename, 0, gt=gt)
        print('Generated image saved at:', pathlib.Path(save_path))

    return in_between


if __name__ == '__main__':
    model = UNetPadded(2, output_channels=1, hidden_channels=64)
    checkpoint_path = 'checkpoints/gen_checkpoint.pth'
    # input_triplet_path is the argument passed to the script (if any is passed)
    if len(sys.argv) > 1 and sys.argv[1] == 'gt':
        gt = True
        input_triplet_path = 'to_generate/w_gt'
        save_path = 'to_generate/w_gt'
    else:
        gt = False
        input_triplet_path = 'to_generate/wo_gt'
        save_path = 'to_generate/wo_gt'
    generate(model, checkpoint_path, input_triplet_path, save_path=save_path, binary_thresh=0.5, gt=gt)