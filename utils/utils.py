import numpy as np
import math
import cv2

import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


# import utils.calculator as cal

def create_gif(input1, labels, input2, pred, experiment_dir, epoch):
    pred = Image.fromarray((pred.detach().cpu().numpy().squeeze() * 255).astype(np.uint8))
    input1 = Image.fromarray((input1.detach().cpu().numpy().squeeze() * 255).astype(np.uint8))
    input2 = Image.fromarray((input2.detach().cpu().numpy().squeeze() * 255).astype(np.uint8))
    labels = Image.fromarray((labels.detach().cpu().numpy().squeeze() * 255).astype(np.uint8))
    
    # Gif for generated triplet
    input1.save(experiment_dir + 'triplet_' + str(epoch) + 'true_.gif', save_all=True, append_images=[labels, input2], duration=500, loop=0)
    # Gif for ground truth triplet
    input1.save(experiment_dir + 'triplet_' + str(epoch) + 'pred_.gif', save_all=True, append_images=[pred, input2], duration=500, loop=0)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def show_tensor_images(image_tensor, num_images=16):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())


def create_gif(input1, labels, input2, pred, experiment_dir, epoch):
    pred = Image.fromarray(np.squeeze((pred[0].detach().cpu().numpy() * 255), axis=0))
    input1 = Image.fromarray(np.squeeze((input1[0].detach().cpu().numpy() * 255), axis=0))
    input2 = Image.fromarray(np.squeeze((input2[0].detach().cpu().numpy() * 255), axis=0))
    labels = Image.fromarray(np.squeeze((labels[0].detach().cpu().numpy() * 255), axis=0))
    # Gif for generated triplet
    input1.save(experiment_dir + 'triplet_' + str(epoch) + '_true.gif', save_all=True, append_images=[labels, input2], duration=500, loop=0)
    # Gif for ground truth triplet
    input1.save(experiment_dir + 'triplet_' + str(epoch) + '_pred.gif', save_all=True, append_images=[pred, input2], duration=500, loop=0)


def visualize_batch(input1, labels, input2, pred, epoch, experiment_dir='exp/', train_gen_losses=None, train_disc_losses=None,
                    test_gen_losses=None, test_disc_losses=None, figsize=(20,10)):
        
        # Creates experiment directory if it doesn't exist'
        experiment_dir = experiment_dir + 'losses/'
        if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

        if train_gen_losses is not None and test_gen_losses is not None:
            # Plots Generator and Discriminator losses in the same plot
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(train_gen_losses, label='Generetor')
            plt.plot(train_disc_losses, label='Discriminator')
            plt.title("Training Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.subplot(1,2,2)
            plt.plot(test_gen_losses, label='Generetor')
            plt.plot(test_disc_losses, label='Discriminator')
            plt.title("Testing Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(experiment_dir + 'loss' + str(epoch) + '.png')
        
