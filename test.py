from torch.utils.data import DataLoader
import tqdm
from torch import nn
import torch
from loss import get_gen_loss
from torchvision.utils import save_image
from utils.utils import create_gif

# testing function
def test(dataset, gen, disc, adv_l, adv_lambda, epoch, display_step=10, r1=nn.BCELoss(), lambr1=0.5, batch_size=12, 
         device='cuda', experiment_dir='exp/'):
    '''
    Tests a single epoch
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    for input1, real, input2 in tqdm.tqdm(dataloader):
        input1, real, input2 = input1.to(device), real.to(device), input2.to(device)

        preds = gen(input1, input2)

        # print('TEST - input1', input1.shape, 'input2', input2.shape, 'pred', preds.shape, 'labels', real.shape)

        gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, r1=r1, lambr1=lambr1, device=device)

        '''Train discriminator'''
        # Discriminator loss for predicted images
        disc_pred_hat = disc(preds.detach())
        disc_fake_loss = adv_l(disc_pred_hat, torch.zeros_like(disc_pred_hat))
        # Discriminator loss for real images
        disc_real_hat = disc(real)
        disc_real_loss = adv_l(disc_real_hat, torch.ones_like(disc_real_hat))
        # Total discriminator loss
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        
        gen_epoch_loss += gen_loss.item()
        disc_epoch_loss += disc_loss.item()

    if epoch % display_step == 0:
        # Saves torch image with the batch of predicted and real images
        save_image(real, experiment_dir + 'batch_' + str(epoch) + '_real.png', nrow=4, normalize=True)
        save_image(preds, experiment_dir + 'batch_' + str(epoch) + '_preds.png', nrow=4, normalize=True)
        create_gif(input1, real, input2, preds, experiment_dir, epoch) # Saves gifs of the predicted and ground truth triplets

    return gen_epoch_loss/len(dataloader), disc_epoch_loss/len(dataloader)
