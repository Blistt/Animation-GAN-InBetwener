from torch.utils.data import DataLoader
import tqdm
from torch import nn
import torch
from _loss import get_gen_loss
from torchvision.utils import save_image
from _utils.utils import create_gif, write_log
from collections import defaultdict
import numpy as np
from _evaluate import evaluate

# testing function
def test(dataset, gen, disc, adv_l, adv_lambda, epoch, display_step=10, plot_step=10, r1=nn.BCELoss(), 
         lambr1=0.5, r2=None, r3=None, lambr2=None, lambr3=None, metrics=None, batch_size=12, device='cuda', 
         experiment_dir='exp/'):
    '''
    Tests a single epoch
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    step_num = 0
    display_step = len(dataloader)//display_step
    results_e = defaultdict(list)     # Stores internal metrics for an epoch
    
    for input1, real, input2 in tqdm.tqdm(dataloader):
        gen_batch_loss = []
        disc_batch_loss = []

        input1, real, input2 = input1.to(device), real.to(device), input2.to(device)

        '''Get generator loss'''
        preds = gen(input1, input2)
        gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, r1=r1, lambr1=lambr1, 
                                r2=r2, r3=r3, lambr2=lambr2, lambr3=lambr3, device=device)

        '''Get discriminator loss'''
        # Discriminator loss for predicted images
        disc_pred_hat = disc(preds.detach())
        disc_fake_loss = adv_l(disc_pred_hat, torch.zeros_like(disc_pred_hat))
        # Discriminator loss for real images
        disc_real_hat = disc(real)
        disc_real_loss = adv_l(disc_real_hat, torch.ones_like(disc_real_hat))
        # Total discriminator loss
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        
        gen_batch_loss.append(gen_loss.item())
        disc_batch_loss.append(disc_loss.item())

        '''Compute evaluation metrics'''
        if metrics is not None:
            results_e = evaluate(preds, real, metrics, results_e, device)
            
        if step_num % display_step == 0 and epoch % plot_step == 0:
            # Saves torch image with the batch of predicted and real images
            id = str(epoch) + '_' + str(step_num)
            save_image(real, experiment_dir + 'batch_' + id + '_real.png', nrow=4, normalize=True)
            save_image(preds, experiment_dir + 'batch_' + id + '_preds.png', nrow=4, normalize=True)
            create_gif(input1, real, input2, preds, experiment_dir, id) # Saves gifs of the predicted and ground truth triplets

        step_num += 1
    
    gen_epoch_loss = np.mean(gen_batch_loss)
    disc_epoch_loss = np.mean(disc_batch_loss)
        
    return gen_epoch_loss, disc_epoch_loss, results_e
