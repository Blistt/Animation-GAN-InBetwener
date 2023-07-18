import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from generator_crop import UNetCrop
from generator_light import GeneratorLight
from discriminator_crop import DiscriminatorCrop
from discriminator_full import DiscriminatorFull
from utils.utils import weights_init, visualize_batch, create_gif, visualize_batch_eval
from dataset_class import MyDataset
from loss import get_gen_loss
from test import test
import os
from torchvision.utils import save_image
import torchmetrics
import eval.my_metrics as my_metrics
import eval.chamfer_dist as chamfer_dist
from collections import defaultdict


def train(tra_dataset, gen, disc, gen_opt, disc_opt, adv_l, adv_lambda, r1=nn.L1Loss(), lambr1=100, n_epochs=10, 
          batch_size=12, device='cuda:0', metrics=None, display_step=20, test_dataset=None, my_dataset=None, 
        save_checkpoints=True, experiment_dir='exp/'):  
    
    # stores generator losses
    tr_gen_losses = []  
    tr_disc_losses = []
    # stores discriminator losses
    test_gen_losses = []
    test_disc_losses = []
    # Stores metrics
    results = defaultdict(list)
    dataloader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)
    
    '''
    Training Loop
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

            # Aggregates epoch losses
            gen_epoch_loss += gen_loss.item()
            disc_epoch_loss += disc_loss.item()
            

        # '''Train generator again if discriminator loss is twice as low as generator loss'''
        # n_gen_steps = int((gen_loss.item() / (disc_loss.item() * 2)) - 1)   # Calculate number of additional training steps for generator
        # for i in range(n_gen_steps):
        #     # Train generator again
        #     gen_opt.zero_grad()
        #     preds = gen(input1, input2)
        #     gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, r1=r1, device=device, lambr1=lambr1)
        #     gen_loss.backward()
        #     gen_opt.step()

        # Aggregates losses so far
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
                test_gen_loss, test_disc_loss, epoch_metrics = test(test_dataset, gen, disc, adv_l, adv_lambda, epoch, r1=r1,
                                                     lambr1=lambr1, batch_size=batch_size, metrics=metrics, device=device,
                                                     experiment_dir=experiment_dir+'test/')
            # Aggregates losses and metrics so far
            test_gen_losses.append(test_gen_loss)
            test_disc_losses.append(test_disc_loss)
            for key, value in epoch_metrics.items():
                results[key].append(value.item())
            
       
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
            save_image(real, train_dir + str(epoch) + '_real.png', nrow=4, normalize=True)
            save_image(preds, train_dir + str(epoch) + '_preds.png', nrow=4, normalize=True)
            create_gif(input1, real, input2, preds, experiment_dir+'train/', epoch) # Saves gifs of the predicted and ground truth triplets

            # Saves checkpoing with model's current state
            if save_checkpoints:
                torch.save(gen.state_dict(), experiment_dir + 'gen_checkpoint' + str(epoch) + '.pth')
                torch.save(disc.state_dict(), experiment_dir + 'disc_checkpoint' + str(epoch) + '.pth')

            # Plots metrics
            visualize_batch_eval(results, epoch, experiment_dir=experiment_dir, train_test='test')

        
        print(f"Epoch {epoch}: Training Gen loss: {tr_gen_losses[-1]} Training Disc loss: {tr_disc_losses[-1]} "
        f"Testing Gen loss: {test_gen_losses[-1]} Testing Disc loss: {test_disc_losses[-1]}")
        
        # Keeps a log of the training and testing losses
        with open(experiment_dir + 'training_log.txt', 'a') as f:
            print(f"Epoch {epoch}: Training Gen loss: {tr_gen_losses[-1]} Training Disc loss: {tr_disc_losses[-1]} "
                f"Testing Gen loss: {test_gen_losses[-1]} Testing Disc loss: {test_disc_losses[-1]}", file=f)