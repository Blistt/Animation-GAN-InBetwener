from _utils import get_gen_loss, get_disc_loss
import torch


def gen_train_extra(input1, input2, real, gen, gen_extra, gen_opt, tr_gen_losses, disc, disc_loss, tr_disc_losses, adv_l, adv_lambda, r1, lambr1, 
                    r2=None, lambr2=None, r3=None, lambr3=None, device='cuda:0'):
    '''
    Trains generator again if the discriminator loss is improving faster than the generator loss
    '''
    if len(tr_gen_losses) > 1 and len(tr_disc_losses) > 1:      # Runs only if there are at least 2 generator and discriminator losses
        # Calculate exponential averages of generator and discriminator losses
        gen_exp_average = (0.9 * tr_gen_losses[-2]) - (0.1 * tr_gen_losses[-1]) 
        disc_exp_average = (0.9 * tr_disc_losses[-2]) - (0.1 * tr_disc_losses[-1]) 

        # Calculates normalized improvement of generator and discriminator losses over their exponential average
        disc_improvement = (disc_exp_average - disc_loss.item()) / disc_exp_average
        gen_improvement = (gen_exp_average - gen_loss.item()) / gen_exp_average
        
        # Calculates number of additional training steps for generator (limits to "gen_extra" number of extra steps)
        n_gen_steps = min(gen_extra, int((disc_improvement/ (gen_improvement)) - 1)) 
        if n_gen_steps > 0:  
            print('Number of additional training steps for generator: ' + str(n_gen_steps))
            for i in range(n_gen_steps):
                # Train generator again
                gen_opt.zero_grad()
                preds = gen(input1, input2)
                gen_loss = get_gen_loss(preds, disc, real, adv_l, adv_lambda, r1=r1, r2=r2, r3=r3, 
                                        lambr1=lambr1, lambr2=lambr2, lambr3=lambr3, device=device)
                gen_loss.backward()
                gen_opt.step()
    return gen_loss


def disc_train_extra(preds, real, disc, adv_l, disc_extra, disc_opt, disc_loss, tr_disc_losses, gen_loss, tr_gen_losses):
    '''
    Trains discriminator again if generator loss is twice as low as discriminator loss
    '''
    if len(tr_gen_losses) > 1 and len(tr_disc_losses) > 1:      # Runs only if there are at least 2 generator and discriminator losses
        # Calculate exponential averages of generator and discriminator losses
        gen_exp_average = (0.9 * tr_gen_losses[-2]) - (0.1 * tr_gen_losses[-1]) 
        disc_exp_average = (0.9 * tr_disc_losses[-2]) - (0.1 * tr_disc_losses[-1]) 

        # Calculates normalized improvement of generator and discriminator losses over their exponential average
        disc_improvement = (disc_exp_average - disc_loss.item()) / disc_exp_average
        gen_improvement = (gen_exp_average - gen_loss.item()) / gen_exp_average
        
        # Calculates number of additional training steps for generator (limits to "gen_extra" number of extra steps)
        n_disc_steps = min(disc_extra, int((gen_improvement/ (disc_improvement)) - 1)) 
        if n_disc_steps > 0:
            print('Number of additional training steps for discriminator: ' + str(n_disc_steps))
            for i in range(n_disc_steps):
                # Train discriminator again
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
    return disc_loss
