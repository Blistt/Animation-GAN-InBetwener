import torch
from torch import nn
from torchvision import transforms
from generator_full import UNetFull
from discriminator_full import DiscriminatorFull
from utils.utils import weights_init
from dataset_class import MyDataset
import os
import torchmetrics
import eval.my_metrics as my_metrics
import eval.chamfer_dist as chamfer_dist
from train import train


if __name__ == '__main__':
    
    device = 'cuda:1'

    '''Loss function parameters'''
    adv_l = nn.BCEWithLogitsLoss().to(device)    # Adversarial loss
    recon_l = nn.L1Loss()                   # Reconstruction loss 1
    # gdl_l = GDL(device)                   # Reconstruction loss 2
    # ms_ssim_l = MS_SSIM(device)         # Reconstruction loss 3
    adv_lambda = 1.0                 # Adversarial loss weight
    recon_lambda = 1.0                  # Reconstruction loss 1 weight        


    '''Training loop parameters'''
    n_epochs = 100                      # Number of epochs
    input_dim = 2                       # Input channels (1 for each grayscale input frame)
    label_dim = 1                       # Output channels (1 for each grayscale output frame)
    hidden_channels = 64                # Hidden channels of the generator and discriminator
    display_step = 6                   # How often to display/visualize the images
    batch_size = 6                     # Batch size
    lr = 0.0002                         # Learning rate
    b1 = 0.5                            # Adam: decay of first order momentum of gradient
    b2 = 0.999                          # Adam: decay of second order momentum of gradient
    img_size = (512, 512)                      # Frames' image size
    target_size = (373, 373)                   # Cropped frames' image size


    '''Model parameters'''
    gen = UNetFull(input_dim, label_dim, use_bn=True, use_dropout=True).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(b1, b2))
    disc = DiscriminatorFull(label_dim, hidden_channels).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(b1, b2))


    '''Dataset parameters'''
    transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Resize(img_size, antialias=True),])
    binary_threshold = 0.75
    # Training dataset
    # train_data_dir = 'mini_datasets/mini_train_triplets/'
    train_data_dir = '/data/farriaga/atd_12k/Line_Art/train_10k/'
    train_dataset = MyDataset(train_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold)
    # Testing dataset (optional)
    # test_data_dir = 'mini_datasets/mini_test_triplets/'
    test_data_dir = '/data/farriaga/atd_12k/Line_Art/test_2k_original/'
    test_dataset = MyDataset(test_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold)
    # MY dataset (optional)
    my_data_dir = 'mini_datasets/mini_real_test_triplets/'
    my_dataset = MyDataset(my_data_dir, transform=transform, resize_to=img_size, binarize_at=binary_threshold)
    

    '''
    Evaluation parameters
    '''
    other_device = 'cuda:1' if device == torch.device('cuda:0') else 'cuda:0'
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
    experiment_dir = 'exp1_full/'
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    # Loads pre-trained model if specified
    pretrained = False
    if pretrained:
        loaded_state = torch.load("pix2pix_15000.pth")
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
        disc.load_state_dict(loaded_state["disc"])
        disc_opt.load_state_dict(loaded_state["disc_opt"])
    else:
        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init)

    train(train_dataset, gen, disc, gen_opt, disc_opt, adv_l, adv_lambda, r1=recon_l, lambr1=recon_lambda, n_epochs=n_epochs, 
          batch_size=batch_size, device=device, metrics=metrics, display_step=display_step, test_dataset=test_dataset, my_dataset=my_dataset, 
          experiment_dir=experiment_dir)