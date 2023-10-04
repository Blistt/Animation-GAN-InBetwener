"""
This script generates an image using a chosen model and a checkpoint.
"""


import os
import torch


def generate(gen, input_path, output_path):
    """
    Generates an image using a chosen model and a checkpoint
    param gen: Generator model
    param input_path: Path to the input image
    param output_path: Path to the output image
    """

    input = torch.load(input_path)
    # Generates output image
    output = gen(input)
    # Saves output image
    torch.save(output, output_path)