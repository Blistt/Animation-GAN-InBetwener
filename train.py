import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from generator_light import GeneratorLight
from discriminator import Discriminator
from utils import weights_init
from dataset_class import MyDataset
from loss import get_gen_loss, gdl_loss, MS_SSIM


def train(tra_dataset, gen, disc, gen_opt, disc_opt, adv_l, adv_lambda, l1=nn.L1Loss(), l2=None, l3=None, 
          lamb1=100, lamb2=100, lamb3=100, n_epochs=10, batch_size=12, device='cuda:1', test_dataset=None, experiment_dir='exp/'):  
    
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

        # Stores losses for display purposes
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        for input1, real, input2 in tqdm(dataloader):
            input1 = input1.to(device)
            real = real.to(device)
            input2 = input2.to(device)

            '''Train generator'''
            gen_opt.zero_grad()
            preds = gen(input1, input2)
            gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, l1=l1, l2=l2, l3=l3, 
                                    lamb1=lam1, lamb2=lam2, lamb3=lam3)
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
            torch.cuda.empty_cache()    # Free up unused memory before starting testing process
            gen.eval(), disc.eval()     # Set the model to evaluation mode
            args = locals()
            # Evaluate the model on the test dataset
            with torch.no_grad():
                test_gen_loss, test_disc_loss = test(gen, disc, adv_l, adv_lambda, l1=l1, l2=l2, l3=l3, 
                                                     lamb1=lamb1, lamb2=lamb2, lamb3=lamb3, batch_size=batch_size, device='cuda:0',
                                                     test_dataset=test_dataset, experiment_dir=experiment_dir)
            test_gen_losses.append(test_gen_loss)
            test_disc_losses.append(test_disc_loss)
            
            
if __name__ == '__main__':
    '''Loss function parameters'''
    adv_l = nn.BCEWithLogitsLoss()
    recon_l = nn.L1Loss()
    gdl_l = gdl_loss
    ms_ssim_l = MS_SSIM()
    adv_lambda = 200
    recon_lambda = 200
    gdl_lambda = 200
    ms_ssim_lambda = 200

    '''Training loop parameters'''
    n_epochs = 20
    input_dim = 3
    real_dim = 3
    display_step = 200
    batch_size = 4
    lr = 0.0002
    target_shape = 256
    device = 'cuda'

    '''Dataset parameters'''
    transform = transforms.Compose([transforms.ToTensor(),])
    # Training dataset
    training_data_dir = "data/frames"
    tra_dataset = MyDataset(training_data_dir, transform=transform)

    # Testing dataset (optional)
    testing_data_dir = "data/frames"
    test_dataset = MyDataset(testing_data_dir, transform=transform)

    '''Model parameters'''
    gen = GeneratorLight(input_dim, real_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator(input_dim + real_dim).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


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

train()