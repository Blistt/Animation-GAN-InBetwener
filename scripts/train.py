import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.utils import weights_init, visualize_batch_loss_fit, visualize_batch_loss_gan, create_gif, visualize_batch_eval
from support.loss import get_gen_loss
from scripts.test import test
import os
from torchvision.utils import save_image
from collections import defaultdict
import numpy as np
from utils.utils import get_edt
from eval.evaluate import evaluate


def train(tra_dataset, gen, disc, gen_opt, disc_opt, adv_l, adv_lambda, r1=nn.L1Loss(), lambr1=1.0, 
          r2=None, r3=None, lambr2=None, lambr3=None, n_epochs=10, batch_size=12, device='cuda:0', 
          metrics=None, display_step=4, plot_step=10, val_dataset=None, test_dataset=None, save_checkpoints=True, 
          experiment_dir='exp/'):  
    
    # Prints all function parameters in experiment directory
    with open(experiment_dir + 'parameters.txt', 'w') as f:
        for param in locals().items():
            print(param, file=f)
    
    # Creates directory to store training results
    train_dir = experiment_dir + 'train/'
    os.makedirs(train_dir, exist_ok=True)

    
    # stores generator losses
    tr_gen_losses = []  
    tr_disc_losses = []
    # stores discriminator losses
    test_gen_losses = []
    test_disc_losses = []
    # Stores metrics
    train_results_epoch = defaultdict(list)
    test_results_epoch = defaultdict(list)
    
    dataloader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)
    train_display_step = len(dataloader)//display_step

    '''
    ######################## TRAINING LOOP ############################
    '''
    step_num = 0
    for epoch in range(n_epochs):
        gen_loss_batch = []
        disc_loss_batch = []
     
        print('Epoch: ' + str(epoch))
        gen.train(), disc.train()       # Set the models to training mode
        train_results_e = defaultdict(list)     # Stores internal metrics for an epoch

        for input1, real, input2 in tqdm(dataloader):
            input1, real, input2 = input1.to(device), real.to(device), input2.to(device)

            '''Train discriminator'''
            disc_opt.zero_grad()
            # Discriminator loss for predicted images
            with torch.no_grad():
                preds = gen(input1, input2)
            disc_pred_hat = disc(preds.detach())
            disc_fake_loss = adv_l(disc_pred_hat, torch.zeros_like(disc_pred_hat))
            # Discriminator loss for real images
            disc_real_hat = disc(real)
            disc_real_loss = adv_l(disc_real_hat, torch.ones_like(disc_real_hat))
            # Total discriminator loss
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            '''Train generator'''
            gen_opt.zero_grad()
            preds = gen(input1, input2)
            gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, r1=r1, r2=r2, r3=r3, 
                                    lambr1=lambr1, lambr2=lambr2, lambr3=lambr3, device=device)
            gen_loss.backward()
            gen_opt.step()

            # Stores batch losses
            disc_loss_batch.append(disc_loss.item())
            gen_loss_batch.append(gen_loss.item())

            '''Evaluate metrics on training batch'''
            train_results_e = evaluate(preds, real, metrics, train_results_e, device)

            '''Visualizes predictions'''
            if step_num % train_display_step == 0 and epoch % plot_step == 0:            
                # Saves torch image with the batch of predicted and real images
                save_image(real, train_dir + str(step_num) + '_real.png', nrow=4, normalize=False)
                save_image(preds, train_dir + str(step_num) + '_preds.png', nrow=4, normalize=False)
                create_gif(input1, real, input2, preds, experiment_dir+'train/', step_num) # Saves gifs of the predicted and ground truth triplets
            
            step_num += 1
        
        # Appends mean batch losses
        tr_gen_losses.append(np.mean(gen_loss_batch))
        tr_disc_losses.append(np.mean(disc_loss_batch))


        '''
        ######################## VALIDATION ############################
        '''
        os.makedirs(experiment_dir+'test/', exist_ok=True)
        torch.cuda.empty_cache()    # Free up unused memory before starting testing process
        gen.eval(), disc.eval()     # Set the model to evaluation mode
        '''Evaluate the model on the validation dataset'''
        with torch.no_grad():
            test_gen_loss, test_disc_loss, test_results_e = test(val_dataset, gen, disc, adv_l, adv_lambda, epoch, display_step=display_step, 
                                                                        plot_step=plot_step, r1=r1, lambr1=lambr1, r2=r2, r3=r3, 
                                                                        lambr2=lambr2, lambr3=lambr3, batch_size=batch_size, 
                                                                        metrics=metrics, device=device, 
                                                                        experiment_dir=experiment_dir+'test/')
        
        
        '''Performs testing in Testing dataset'''
        if test_dataset is not None:
            my_display_step = 1
            os.makedirs(experiment_dir+'cool_test/', exist_ok=True)
            # Free up unused memory before starting testing process
            torch.cuda.empty_cache()
            gen.eval(), disc.eval()     # Set the model to evaluation mode              
            with torch.no_grad():
                unused_loss = test(test_dataset, gen, disc, adv_l, adv_lambda, epoch, display_step=my_display_step, plot_step=plot_step, 
                                   r1=r1, lambr1=lambr1, r2=r2, r3=r3, lambr2=lambr2, lambr3=lambr3, batch_size=batch_size, 
                                    device=device, experiment_dir=experiment_dir+'cool_test/')
            
       
        '''
        ################## PLOTS AND CHECKPOINTS ##################
        '''
        # Aggregates test losses for the whole epoch (these are lists, so addition means appending)
        test_gen_losses.append(test_gen_loss)
        test_disc_losses.append(test_disc_loss)
        # Aggregates test and training metrics for the whole epoch
        for metric in metrics:
            test_results_epoch[metric].append(np.mean(test_results_e[metric]))
            train_results_epoch[metric].append(np.mean(train_results_e[metric]))

        # Saves checkpoing with model's current state
        if save_checkpoints:
            if epoch > 0:
                # Only save checkpoint if reaching the minimum chamfer metric
                if test_results_epoch['chamfer'][-1] <= min(test_results_epoch['chamfer']):
                    torch.save(gen.state_dict(), experiment_dir + 'gen_checkpoint.pth')
                    torch.save(disc.state_dict(), experiment_dir + 'disc_checkpoint.pth')

        if epoch % plot_step == 0:
            # Save snapshot of model architecture``
            if epoch == 0:
                with open(experiment_dir + 'gen_architecture.txt', 'w') as f:
                    print(gen, file=f)
                with open(experiment_dir + 'disc_architecture.txt', 'w') as f:
                    print(disc, file=f)

            # Plots losses
            visualize_batch_loss_fit(experiment_dir=experiment_dir, train_gen_losses=tr_gen_losses,
                            train_disc_losses=tr_disc_losses, test_gen_losses=test_gen_losses, test_disc_losses=test_disc_losses)
            visualize_batch_loss_gan(experiment_dir=experiment_dir, train_gen_losses=tr_gen_losses,
                            train_disc_losses=tr_disc_losses, test_gen_losses=test_gen_losses, test_disc_losses=test_disc_losses)

            # Plots metrics
            visualize_batch_eval(test_results_epoch, train_results_epoch, experiment_dir=experiment_dir, train_test='metrics')

            # Prints losses
            print(f"Epoch {epoch}: Training Gen loss: {tr_gen_losses[-1]} Training Disc loss: {tr_disc_losses[-1]} "
            f"Testing Gen loss: {test_gen_losses[-1]} Testing Disc loss: {test_disc_losses[-1]}")
        
            # Keeps a written log of the training and testing losses
            with open(experiment_dir + 'training_log.txt', 'a') as f:
                print(f"Epoch {epoch}: Training Gen loss: {tr_gen_losses[-1]} Training Disc loss: {tr_disc_losses[-1]} "
                    f"Testing Gen loss: {test_gen_losses[-1]} Testing Disc loss: {test_disc_losses[-1]}", file=f)
        