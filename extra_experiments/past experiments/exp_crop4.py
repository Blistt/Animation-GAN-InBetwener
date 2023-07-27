import torch
from torch import nn
from torchvision import transforms
from generator_crop import UNetCrop
from generator_light import GeneratorLight
from discriminator_crop import DiscriminatorCrop
from discriminator_full import DiscriminatorFull
from utils.utils import weights_init
from dataset_class import MyDataset
import os
import torchmetrics
import eval.my_metrics as my_metrics
import eval.chamfer_dist as chamfer_dist
from train import train
from loss import GDL, MS_SSIM


if __name__ == '__main__':
    
    device = 'cuda:1'

    '''Loss function parameters'''
    adv_l = nn.BCEWithLogitsLoss().to(device)    # Adversarial loss
    r1 = nn.L1Loss().to(device)             # Reconstruction loss 1
    # r2 = GDL(device)                   # Reconstruction loss 2
    # r3 = MS_SSIM(device)            # Reconstruction loss 3
    r2=None
    r3=None
    adv_lambda = 0.05                 # Adversarial loss weight
    r1_lambda = 4.0                  # Reconstruction loss 1 weight        
    r2_lambda = 1.0                  # Reconstruction loss 2 weight
    r3_lambda = 6.0                  # Reconstruction loss 3 weight

    '''Training loop parameters'''
    n_epochs = 501                      # Number of epochs
    input_dim = 2                       # Input channels (1 for each grayscale input frame)
    label_dim = 1                       # Output channels (1 for each grayscale output frame)
    hidden_channels = 64                # Hidden channels of the generator and discriminator
    display_step = 6                   # How often to display/visualize the images
    batch_size = 4                     # Batch size
    lr = 0.0002                         # Learning rate
    b1 = 0.5                            # Adam: decay of first order momentum of gradient
    b2 = 0.999                          # Adam: decay of second order momentum of gradient
    img_size = (512, 512)                      # Frames' image size
    target_size = (373, 373)                   # Cropped frames' image size


    '''Model parameters'''
    gen = UNetCrop(input_dim, label_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(b1, b2))
    disc = DiscriminatorCrop(label_dim, hidden_channels).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(b1, b2))
    save_checkpoints = False


    '''Dataset parameters'''
    transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Resize(img_size, antialias=True),])
    binary_threshold = 0.75
    # Training dataset
    train_data_dir = 'mini_datasets/mini_train_triplets/'
    # train_data_dir = '/data/farriaga/atd_12k/Line_Art/train_10k/'
    train_dataset = MyDataset(train_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                               crop_shape=target_size)
    # Testing dataset (optional)
    test_data_dir = 'mini_datasets/mini_test_triplets/'
    # test_data_dir = '/data/farriaga/atd_12k/Line_Art/test_2k_original/'
    test_dataset = MyDataset(test_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                             crop_shape=target_size)
    # MY dataset (optional)
    my_data_dir = 'mini_datasets/mini_real_test_triplets/'
    my_dataset = MyDataset(my_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold,
                           crop_shape=target_size)
    

    '''
    Evaluation parameters
    '''
    other_device = 'cuda:1' if device == 'cuda:0' else 'cuda:0'
    metrics = torchmetrics.MetricCollection({
        'psnr': my_metrics.PSNRMetricCPU(),
        'ssim': my_metrics.SSIMMetricCPU(),
        'chamfer': chamfer_dist.ChamferDistance2dMetric(binary=0.5),
        'mse': torchmetrics.MeanSquaredError(),
    }).to(other_device).eval()



    '''
    Visualization parameters
    '''
    display_step = 1
    plot_step = 20
    experiment_dir = 'exp4_crop_mini/'
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    # Loads pre-trained model if specified
    pretrained = True
    if pretrained:
        loaded_state = torch.load('/data/farriaga/Experiments/unet_int/exp3/checkpoint10.pth')
        gen.load_state_dict(loaded_state)
    else:
        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init)

    # Records time it takes to train the model
    import time
    start_time = time.time()

    train(train_dataset, gen, disc, gen_opt, disc_opt, adv_l, adv_lambda, r1=r1, lambr1=r1_lambda, 
          r2=r2, r3=r3, lambr2=r2_lambda, lambr3=r3_lambda, n_epochs=n_epochs, batch_size=batch_size, 
          device=device, metrics=metrics, display_step=display_step, plot_step=plot_step, test_dataset=test_dataset,
          my_dataset=my_dataset, save_checkpoints=save_checkpoints, experiment_dir=experiment_dir)
    
    # Saves the time it took in a text file in experiment directory
    with open(experiment_dir + 'time.txt', 'w') as f:
        f.write(f'Training took {(time.time() - start_time)/60} minutes')