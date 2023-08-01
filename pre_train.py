from torch.utils.data import DataLoader
import tqdm
from loss import pre_train_loss
from torchvision.utils import save_image
from utils.utils import create_gif
import os


def pre_train(model, model_opt, tra_dataset, r1, lambr1, r2=None, r3=None, lambr2=None, lambr3=None,
              n_epochs=1, batch_size=10, device='cuda', train_dir='pre_train/', experiment_dir='temp/'):
    print('Pre-training generator')

    # Initializes directory to save images
    experiment_dir = experiment_dir + train_dir
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    '''
    Training loop
    '''
    cur_step = 0
    dataloader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)
    step_num = 0

    for epoch in range(n_epochs):
        print('Epoch: ' + str(epoch))
        model.train()       # Set the model to training mode
        epoch_loss = 0
        for input1, real, input2 in tqdm.tqdm(dataloader):
            # Flatten the image
            input1, real, input2 = input1.to(device), real.to(device), input2.to(device)

            model_opt.zero_grad()
            pred = model(input1, input2)
            model_loss = pre_train_loss(pred, real, r1, lambr1, r2=r2, r3=r3, lambr2=lambr2, lambr3=lambr3, device=device)
            model_loss.backward()
            model_opt.step()
            cur_step += 1

            save_image(real, experiment_dir + str(step_num) + '_real.png', nrow=4, normalize=True)
            save_image(pred, experiment_dir + str(step_num) + '_preds.png', nrow=4, normalize=True)
            create_gif(input1, real, input2, pred, experiment_dir, step_num) 
            step_num += 1

    return model
