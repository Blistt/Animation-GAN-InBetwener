import numpy as np
import math
import cv2

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageSequence
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import csv
from eval.chamfer_dist import batch_edt
import imageio


def get_edt(imgs, exp_factor=540/25, binarize_at=0.5, bit_reverse=True, device='cuda:0'):
    # Transfer tensors to other device to avoid issues with memory leak
    other_device = 'cuda:1' if device == 'cuda:0' else 'cuda:0'
    imgs = imgs.to(other_device)

    # Binarize images
    imgs = (imgs>binarize_at).float()

    if bit_reverse:
        imgs = 1-imgs

    # Calculate NEDT
    edt = batch_edt(imgs)
    nedt = 1 - (-edt*exp_factor / max(edt.shape[-2:])).exp()
    
    return nedt


def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''

    # transforms image to torch tensor if it is PIL Image
    if isinstance(image, Image.Image):
        image = ToTensor()(image)

    padding_y = (image.shape[-2]-new_shape[-2]) // 2
    padding_x = (image.shape[-1]-new_shape[-1]) // 2
    odd_y, odd_x = (image.shape[-2]-new_shape[-2]) % 2, (image.shape[-1]-new_shape[-1]) % 2
    # Crops whole batch or a single image
    if image.dim() > 3:
      cropped_image = image[:, :, padding_y+odd_y:image.shape[-2]-padding_y, padding_x+odd_x:image.shape[-1]-padding_x]
    else:
      cropped_image = image[:, padding_y+odd_y:image.shape[-2]-padding_y, padding_x+odd_x:image.shape[-1]-padding_x]
    return cropped_image

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)


def siren_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n_in = m.in_channels
        n_units = m.out_channels
        c = 1 / math.sqrt(n_in) * math.sqrt(6 / (n_in + n_units))
        nn.init.uniform_(m.weight, -c, c)


def show_tensor_images(image_tensor, num_images=16):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())


def create_gif(input1, labels, input2, pred, experiment_dir, epoch, stand_alone=False):

    # Splits pred image into RGB channels
    pred_label = pred.repeat(1, 3, 1, 1).reshape(pred.shape[0], 3, pred.shape[2], pred.shape[3])
    # Shows labels as red pixels by setting the intensity of the B and G channels to 0 when the label is 0 (a black pixel)
    pred_label[:, 1, :, :] = pred_label[:, 1, :, :] * labels.squeeze(dim=1)
    pred_label[:, 2, :, :] = pred_label[:, 2, :, :] * labels.squeeze(dim=1)

    input1, input2 = crop(input1[0], pred.shape), crop(input2[0], pred.shape)
    pred = Image.fromarray(np.squeeze((pred[0].detach().cpu().numpy() * 255), axis=0))
    input1 = Image.fromarray(np.squeeze((input1.detach().cpu().numpy() * 255), axis=0))
    input2 = Image.fromarray(np.squeeze((input2.detach().cpu().numpy() * 255), axis=0))
    labels = Image.fromarray(np.squeeze((labels[0].detach().cpu().numpy() * 255), axis=0))
    pred_label = Image.fromarray((pred_label[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8'))
  
    # Gif for generated triplet
    input1.save(experiment_dir + 'triplet_' + str(epoch) + '_true.gif', save_all=True, append_images=[labels, input2], 
                duration=500, loop=0)
    if stand_alone==False:
        # Gif for ground truth triplet
        input1.save(experiment_dir + 'triplet_' + str(epoch) + '_pred.gif', save_all=True, append_images=[pred, input2], 
                    duration=500, loop=0)
        # Gif for generated triplet with label
        input1.save(experiment_dir + 'triplet_' + str(epoch) + '_pred_label.gif', save_all=True, append_images=[pred_label, input2],
                    duration=500, loop=0)


def visualize_batch_loss_gan(experiment_dir='exp/', train_gen_losses=None, train_disc_losses=None,
                    test_gen_losses=None, test_disc_losses=None, figsize=(20,10)):
        
        # Creates experiment directory if it doesn't exist'
        experiment_dir = experiment_dir + 'losses/'
        if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

        if train_gen_losses is not None and test_gen_losses is not None:
            # Plots Generator and Discriminator losses in the same plot
            plt.figure(figsize=figsize)
            plt.subplot(1,2,1)
            plt.plot(train_gen_losses, label='Generetor')
            plt.plot(train_disc_losses, label='Discriminator')
            plt.title("Training Loss per Epoch")
            plt.xlabel("Training step")
            plt.ylabel("Loss")
            plt.legend()
            plt.subplot(1,2,2)
            plt.plot(test_gen_losses, label='Generetor')
            plt.plot(test_disc_losses, label='Discriminator')
            plt.title("Testing Loss per Epoch")
            plt.xlabel("Testing step")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(experiment_dir + 'loss_gan.png')
            plt.close()


def visualize_batch_loss_fit(experiment_dir='exp/', train_gen_losses=None, train_disc_losses=None,
                    test_gen_losses=None, test_disc_losses=None, figsize=(20,10)):
        
        # Creates experiment directory if it doesn't exist'
        experiment_dir = experiment_dir + 'losses/'
        if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

        if train_gen_losses is not None and test_gen_losses is not None:
            # Plots Generator and Discriminator losses in the same plot
            plt.figure(figsize=figsize)
            plt.subplot(1,2,1)
            plt.plot(train_gen_losses, label='Train')
            plt.plot(test_gen_losses, label='Test')
            plt.title("Generator loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.subplot(1,2,2)
            plt.plot(train_disc_losses, label='Train')
            plt.plot(test_disc_losses, label='Test')
            plt.title("Discriminator loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(experiment_dir + 'loss_fit.png')
            plt.close()
        

def visualize_batch_eval(test_metrics, train_metrics, experiment_dir='exp/', train_test='metrics', size=(20, 20)):

    # Creates experiment directory if it doesn't exist'
    experiment_dir = experiment_dir + train_test
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    # Plots all metrics in metrics dictionary in a plt figure
    # Create a new figure with multiple subplots arranged in a square grid
    num_metrics = len(test_metrics)
    num_rows = math.ceil(math.sqrt(num_metrics))
    num_cols = math.ceil(num_metrics / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=size)
    
    # Flatten the axes array to make it easier to iterate over
    axes = axes.flatten()
    
    # Plot both train and test values of each metric in a separate subplot
    for i, (metric, values) in enumerate(test_metrics.items()):
        axes[i].plot(values, label='test')
        axes[i].plot(train_metrics[metric], label='train')
        axes[i].set_title(metric)
        axes[i].set_xlabel('Epoch')
        axes[i].legend()

    # Save the plot to a file
    plt.savefig(f'{experiment_dir}/metrics.png')
    
    # Show the plot
    plt.show()
    plt.close()

    write_log(test_metrics, experiment_dir, train_test='test')  # Saves test metrics in a csv file
    write_log(train_metrics, experiment_dir, train_test='train')  # Saves train metrics in a csv file


def write_log(log, experiment_dir, train_test='test'):
    # Open a new file for writing
    with open(experiment_dir + '_' + train_test + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header row
        header = ['index'] + list(log.keys())
        writer.writerow(header)
        
        # Write the data rows
        for i in range(len(log['chamfer'])):
            row = [i] + [log[key][i] for key in log]
            writer.writerow(row)

import imageio
from PIL import Image, ImageDraw

def mark_frame_in_gif(gif_path, frame_index=1):
    # Read the GIF
    gif = imageio.mimread(gif_path)

    # Check if frame_index is valid
    if frame_index < 0 or frame_index >= len(gif):
        raise ValueError("frame_index is out of range")

    # Convert specified frame to PIL Image
    frame = Image.fromarray(gif[frame_index])

    # Create a draw object
    draw = ImageDraw.Draw(frame)

    # Calculate the position of the cross with 10 pixels padding
    cross_start_x = (frame.width - 30, 20)
    cross_end_x = (frame.width - 10, 20)
    cross_start_y = (frame.width - 20, 10)
    cross_end_y = (frame.width - 20, 30)

    # Draw the cross
    draw.line([cross_start_x, cross_end_x], fill='red', width=5)
    draw.line([cross_start_y, cross_end_y], fill='red', width=5)

    # Convert the frame back to numpy array
    gif[frame_index] = np.array(frame)

    # Write the new GIF
    imageio.mimsave(gif_path[:-4] + '_marked.gif', gif, duration=500, loop=0)

    print("Saved marked GIF to", gif_path[:-4] + '_marked.gif')


def crop_gif(input_path, shape=(373,373)):
    gif = Image.open(input_path)

    frames = []
    # Define the CenterCrop transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(shape)
    ])

    for frame in ImageSequence.Iterator(gif):
        # Center crop frame to size of shape with torch transforms
        cropped = transform(frame)
        cropped = cropped / cropped.max()
        cropped = transforms.ToPILImage()(cropped)
        frames.append(cropped)
    
    frames[0].save(input_path[:-4] + '_cropped.gif', save_all=True, append_images=frames[1:], 
                duration=500, loop=0)
    
    print('Cropped gif saved at', input_path[:-4] + '_cropped.gif')
        

