# import wandb
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from data.imca import ConditionalDataset
from .ivae_core import iVAE
from model.utils import EarlyStopper
# from accelerate import Accelerator

def IVAE_wrapper(data_loader, batch_size=256, max_iter=7e4, n_layer=3, lr=1e-3, seed=0, 
                 cuda=True, ckpt_file='ivae.pt', test=False, model=None, data_loader_valid=None):
    " args are the arguments from the main.py file"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # accelerator = Accelerator()
    # device = accelerator.device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # training loop
    if not test:
        # define optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
        
        # model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
        
        model.train()
        for it in range(max_iter):
            elbo_train = 0
            for _, (x, u) in enumerate(data_loader):
                optimizer.zero_grad()
                x, u = x.to(device), u.to(device)
                elbo, z_est = model.elbo(x, u)
                elbo.mul(-1).backward()
                # accelerator.backward(elbo.mul(-1))
                optimizer.step()
                elbo_train += -elbo.item()
            elbo_train /= len(data_loader)
            scheduler.step(elbo_train)

            if data_loader_valid is not None:
                elbo_valid = 0
                for _, (x, u) in enumerate(data_loader_valid):
                    x, u = x.to(device), u.to(device)
                    elbo, z_est = model.elbo(x, u)
                    elbo_valid += -elbo.item()
                elbo_valid /= len(data_loader_valid)
                loss = [elbo_train, elbo_valid]
                # print(f'iVAE training loss: {elbo_train:.3f}; validation loss: {elbo_valid:.3f}')
                # wandb.log({'iVAE training loss': elbo_train, 'iVAE validation loss': elbo_valid})
            else:
                loss = elbo_train
                print(f'iVAE training loss: {elbo_train:.3f}')
                # wandb.log({'iVAE training loss': elbo_train})
        # save model checkpoint after training
        torch.save(model.state_dict(), ckpt_file)
    else:
        model_params = torch.load(ckpt_file, map_location=device)
        with torch.no_grad():
            for l in range(n_layer):
                model.logl.fc[l].weight.copy_(model_params[f'logl.fc.{l}.weight'])
                model.logl.fc[l].bias.copy_(model_params[f'logl.fc.{l}.bias'])
                model.f.fc[l].weight.copy_(model_params[f'f.fc.{l}.weight'])
                model.f.fc[l].bias.copy_(model_params[f'f.fc.{l}.bias'])
                model.g.fc[l].weight.copy_(model_params[f'g.fc.{l}.weight'])
                model.g.fc[l].bias.copy_(model_params[f'g.fc.{l}.bias'])
                model.logv.fc[l].weight.copy_(model_params[f'logv.fc.{l}.weight'])
                model.logv.fc[l].bias.copy_(model_params[f'logv.fc.{l}.bias'])
        elbo_test = 0
        for _, (x, u) in enumerate(data_loader):
            # x, u = x.to(device), u.to(device)
            elbo, z_est = model.elbo(x, u)
            elbo_test += -elbo.item()
        elbo_test /= len(data_loader)
        # print(f'iVAE test loss: {elbo_test:.3f}')
        loss = elbo_test
    
    return model, loss


def IVAE_init_wrapper(X, U, S, model, batch_size=256, mi_ivae=1000, cuda=True):

    # load data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dset = ConditionalDataset(X=X.astype(np.float32), Y=U.astype(np.float32), S=S.astype(np.float32), device=device)
    loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    data_loader = DataLoader(dset, shuffle=False, batch_size=batch_size, **loader_params)

    loss = torch.nn.MSELoss()
    
    # train encoder to approximate initial sources
    norm_const_s = np.max(np.abs(S))
    optimizer_s = optim.Adam(model.parameters(), lr=1e-3)
    scheduler_s = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s, factor=0.1, patience=20, verbose=True)
    early_stopper = EarlyStopper()

    model.train()
    for it in range(mi_ivae):
        loss_total = 0
        for _, (x, u, s) in enumerate(data_loader):
            optimizer_s.zero_grad()
            # x, u, s = x.to(device), u.to(device), s.to(device)
            _, _, rs, _ = model.forward(x, u)
            loss_batch = loss(rs, s) / norm_const_s
            loss_batch.backward()
            optimizer_s.step()
            loss_total += loss_batch.item()
        loss_total /= len(data_loader)
        scheduler_s.step(loss_total)
        print(f'iVAE initialization - source reconstruction - iteration {it}; training loss: {loss_total:.3f}')
        if early_stopper.early_stop(loss_total):
            print(f'Early stopping triggered!')
            break
    
    # train decoder to approximate data
    norm_const_x = np.max(np.abs(X))
    optimizer_x = optim.Adam(model.parameters(), lr=1e-3) #set weight_decay=1e-4 to avoid ill-conditioned weights
    scheduler_x = optim.lr_scheduler.ReduceLROnPlateau(optimizer_x, factor=0.1, patience=20, verbose=True)
    early_stopper = EarlyStopper()

    # freeze encoder
    for param in model.g.parameters():
        param.requires_grad = False

    for it in range(mi_ivae):
        loss_total = 0
        for _, (x, u, s) in enumerate(data_loader):
            optimizer_x.zero_grad()
            # x, u, s = x.to(device), u.to(device), s.to(device)
            (rx, _), _, _, _ = model.forward(x, u)
            loss_batch = loss(rx, x) / norm_const_x
            loss_batch.backward()
            optimizer_x.step()
            loss_total += loss_batch.item()
        loss_total /= len(data_loader)
        scheduler_x.step(loss_total)
        print(f'iVAE initialization - data reconstruction - iteration {it}; iVAE training loss: {loss_total:.3f}')
        if early_stopper.early_stop(loss_total):
            print(f'Early stopping triggered!')
            break
    
    # activate encoder
    for param in model.g.parameters():
        param.requires_grad = True

    return model