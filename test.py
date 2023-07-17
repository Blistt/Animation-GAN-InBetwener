from torch.utils.data import DataLoader
import tqdm
from torch import nn
import torch
from loss import get_gen_loss
from torchvision.utils import save_image
from utils.utils import create_gif, write_log
from collections import defaultdict
import numpy as np

# testing function
def test(dataset, gen, disc, adv_l, adv_lambda, epoch, display_step=10, r1=nn.BCELoss(), lambr1=0.5, metrics=None, batch_size=12, 
         device='cuda', experiment_dir='exp/'):
    '''
    Tests a single epoch
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    metrics_epoch = defaultdict(list)
    for input1, real, input2 in tqdm.tqdm(dataloader):
        input1, real, input2 = input1.to(device), real.to(device), input2.to(device)

        preds = gen(input1, input2)

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

        '''Compute evaluation metrics'''
        if metrics is not None:
            raw_metrics = metrics(preds, real)
            for k, v in raw_metrics.items():
                metrics_epoch[k].append(v.item())
            write_log(metrics_epoch, experiment_dir, 'test')    #Stores the metrics in a log file

            

    if epoch % display_step == 0:
        # Saves torch image with the batch of predicted and real images
        save_image(real, experiment_dir + 'batch_' + str(epoch) + '_real.png', nrow=4, normalize=True)
        save_image(preds, experiment_dir + 'batch_' + str(epoch) + '_preds.png', nrow=4, normalize=True)
        create_gif(input1, real, input2, preds, experiment_dir, epoch) # Saves gifs of the predicted and ground truth triplets

    mean_epoch_metrics = {k: np.mean(metrics_epoch[k]) for k,v in metrics_epoch.items()}

    return gen_epoch_loss/len(dataloader), disc_epoch_loss/len(dataloader), mean_epoch_metrics
