import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from generator_crop import UNetCrop
from generator_light import GeneratorLight
from discriminator_crop import DiscriminatorCrop
from discriminator_full import DiscriminatorFull
from utils.utils import weights_init, visualize_batch, create_gif
from dataset_class import MyDataset
from loss import get_gen_loss
from test import test
import os
from torchvision.utils import save_image


def train(tra_dataset, gen, disc, gen_opt, disc_opt, adv_l, adv_lambda, r1=nn.L1Loss(), lambr1=100, n_epochs=10, 
          batch_size=12, device='cuda:0', display_step=20, test_dataset=None, my_dataset=None, experiment_dir='exp/'):  
    
    # stores generator losses
    tr_gen_losses = []  
    tr_disc_losses = []
    # stores discriminator losses
    test_gen_losses = []
    test_disc_losses = []
    dataloader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)
    

    '''
    Training loop
    '''
    for epoch in range(n_epochs):
        print('Epoch: ' + str(epoch))
        gen.train(), disc.train()       # Set the models to training mode
        gen_epoch_loss = 0              # Stores losses for display purposes
        disc_epoch_loss = 0
        for input1, real, input2 in tqdm(dataloader):
            input1, real, input2 = input1.to(device), real.to(device), input2.to(device)

            '''Train generator'''
            gen_opt.zero_grad()
            preds = gen(input1, input2)
            gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, r1=r1, device=device, lambr1=lambr1)
            gen_loss.backward()
            gen_opt.step()

            '''Train discriminator'''
            disc_opt.zero_grad()
            # Discriminator loss for predicted images
            disc_pred_hat = disc(preds.detach())
            disc_fake_loss = adv_l(disc_pred_hat, torch.zeros_like(disc_pred_hat))
            # Discriminator loss for real images
            disc_real_hat = disc(real)
            disc_real_loss = adv_l(disc_real_hat, torch.ones_like(disc_real_hat))
            # Total discriminator loss
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()
            gen_epoch_loss += gen_loss.item()
            disc_epoch_loss += disc_loss.item()

        tr_gen_losses.append(gen_epoch_loss/len(dataloader))
        tr_disc_losses.append(disc_epoch_loss/len(dataloader))

        
        '''
        Performs testing if specified
        '''
        if test_dataset is not None:
            os.makedirs(experiment_dir+'test/', exist_ok=True)
            torch.cuda.empty_cache()    # Free up unused memory before starting testing process
            gen.eval(), disc.eval()     # Set the model to evaluation mode
            # Evaluate the model on the test dataset
            with torch.no_grad():
                test_gen_loss, test_disc_loss = test(test_dataset, gen, disc, adv_l, adv_lambda, epoch, r1=r1,
                                                     lambr1=lambr1, batch_size=batch_size, device=device,
                                                     experiment_dir=experiment_dir+'test/')
            test_gen_losses.append(test_gen_loss)
            test_disc_losses.append(test_disc_loss)

        
        '''
        Saves checkpoints, visualizes predictions and plots losses
        '''
        if epoch % display_step == 0:
            # Save snapshot of model architecture``
            if epoch == 0:
                with open(experiment_dir + 'gen_architecture.txt', 'w') as f:
                    print(gen, file=f)
                with open(experiment_dir + 'disc_architecture.txt', 'w') as f:
                    print(disc, file=f)
            '''
            Performs testing in MY dataset if specified
            '''
            if my_dataset is not None:
                os.makedirs(experiment_dir+'cool_test/', exist_ok=True)
                # Free up unused memory before starting testing process
                torch.cuda.empty_cache()
                gen.eval(), disc.eval()     # Set the model to evaluation mode              
                # Evaluate the model on the MY dataset
                with torch.no_grad():
                    unused_loss = test(my_dataset, gen, disc, adv_l, adv_lambda, epoch, r1=r1,  lambr1=lambr1,
                                       batch_size=batch_size, device=device, experiment_dir=experiment_dir+'cool_test/')



            '''MAIN VISUALIZATION BLOCK - Visualizes training predictions and plots training and testing losses'''
            train_dir = experiment_dir + 'train/'
            os.makedirs(train_dir, exist_ok=True)
            visualize_batch(input1, real, input2, preds, epoch, experiment_dir=experiment_dir, train_gen_losses=tr_gen_losses,
                            train_disc_losses=tr_disc_losses, test_gen_losses=test_gen_losses, test_disc_losses=test_disc_losses)
            # Saves torch image with the batch of predicted and real images
            save_image(real, train_dir + str(epoch) + '_real.png', nrows=2, normalize=True)
            save_image(preds, train_dir + str(epoch) + '_preds.png', nrows=2, normalize=True)
            create_gif(input1, real, input2, preds, experiment_dir+'train/', epoch) # Saves gifs of the predicted and ground truth triplets

            # Saves checkpoing with model's current state
            torch.save(gen.state_dict(), experiment_dir + 'gen_checkpoint' + str(epoch) + '.pth')
            torch.save(disc.state_dict(), experiment_dir + 'disc_checkpoint' + str(epoch) + '.pth')

        
        # print(f"Epoch {epoch}: Training Gen loss: {tr_gen_losses[-1]} Training Disc loss: {tr_disc_losses[-1]} "
        # f"Testing Gen loss: {test_gen_losses[-1]} Testing Disc loss: {test_disc_losses[-1]}")

                    
        
        
        # Keeps a log of the training and testing losses
        with open(experiment_dir + 'training_log.txt', 'a') as f:
            print(f"Epoch {epoch}: Training Gen loss: {tr_gen_losses[-1]} Training Disc loss: {tr_disc_losses[-1]} "
                f"Testing Gen loss: {test_gen_losses[-1]} Testing Disc loss: {test_disc_losses[-1]}", file=f)
            
                        

            
if __name__ == '__main__':
    
    device = 'cuda:0'

    '''Loss function parameters'''
    adv_l = nn.BCEWithLogitsLoss().to(device)    # Adversarial loss
    recon_l = nn.BCELoss().to(device)           # Reconstruction loss 1
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
    batch_size = 8                     # Batch size
    lr = 0.00000002                         # Learning rate
    b1 = 0.5                            # Adam: decay of first order momentum of gradient
    b2 = 0.999                          # Adam: decay of second order momentum of gradient
    img_size = (512, 512)                      # Frames' image size
    target_size = (373, 373)                   # Cropped frames' image size

    '''Model parameters'''
    gen = GeneratorLight(label_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(b1, b2))
    disc = DiscriminatorFull(label_dim, hidden_channels).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(b1, b2))


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
    Visualization parameters
    '''
    display_step = 10
    experiment_dir = 'exp1/'
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
          batch_size=batch_size, device=device, display_step=display_step, test_dataset=test_dataset, my_dataset=my_dataset, 
          experiment_dir=experiment_dir)