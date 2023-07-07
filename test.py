from utils.utils import visualize_batch
from torch.utils.data import DataLoader
import tqdm
from torch import nn
import torch
from loss import get_gen_loss

# testing function
def test(dataset, gen, disc, adv_l, adv_lambda, l1=nn.L1Loss(), l2=None, l3=None, lamb1=100, lamb2=100, lamb3=100, 
         batch_size=12, device='cuda:1', experiment_dir='exp/'):
    '''
    Tests a single epoch
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    for input1, real, input2 in tqdm.tqdm(dataloader):
        input1 = input1.to(device)
        real = real.to(device)
        input2 = input2.to(device)


        preds = gen(input1, input2)
        gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, l1=l1, l2=l2, l3=l3, 
                                lamb1=lamb1, lamb2=lamb2, lamb3=lamb3, device=device)

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

    return gen_epoch_loss/len(dataloader), disc_epoch_loss/len(dataloader)
