from torch.utils.data import DataLoader
import tqdm
from loss import pre_train_loss


def pre_train(model, model_opt, tra_dataset, r1, lambr1, r2=None, r3=None, lambr2=None, lambr3=None,
              n_epochs=1, batch_size=10, device='cuda'):
    print('Pre-training generator')
    '''
    Training loop
    '''
    cur_step = 0
    dataloader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        print('Epoch: ' + str(epoch))
        model.train()       # Set the model to training mode
        epoch_loss = 0
        for input1, labels, input2 in tqdm.tqdm(dataloader):
            # Flatten the image
            input1, labels, input2 = input1.to(device), labels.to(device), input2.to(device)

            model_opt.zero_grad()
            pred = model(input1, input2)
            model_loss = pre_train_loss(pred, labels, r1, lambr1, r2=r2, r3=r3, lambr2=lambr2, lambr3=lambr3, device=device)
            model_loss.backward()
            model_opt.step()
            cur_step += 1

    return model
