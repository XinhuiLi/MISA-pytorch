import os
import math
import time
import torch
# import wandb
import pickle
import shutil
import numpy as np
import scipy.io as sio
from dataset.dataset import Dataset
from torch.utils.data import DataLoader
from data.imca import generate_synthetic_data, ConditionalDataset
from metrics.mcc import mean_corr_coef, mean_corr_coef_per_segment
from metrics.mmse import MMSE
from model.ivae.ivae_core import iVAE
from model.ivae.ivae_wrapper import IVAE_wrapper, IVAE_init_wrapper
from model.MISAK import MISA
# from model.MISAKinit import MISA
from model.misa_wrapper import MISA_wrapper_
from model.icebeem_wrapper import ICEBEEM_wrapper_
from model.multiviewica import multiviewica
from data.utils import to_one_hot


def split_sim_data(x, y, s, n_segment, n_obs_per_seg):
    ind_list_train, ind_list_valid, ind_list_test = [], [], []
    for i in range(n_segment):
        ind_list_train += np.arange(i*n_obs_per_seg*3, i*n_obs_per_seg*3+n_obs_per_seg).tolist()
        ind_list_valid += np.arange(i*n_obs_per_seg*3+n_obs_per_seg, i*n_obs_per_seg*3+2*n_obs_per_seg).tolist()
        ind_list_test += np.arange(i*n_obs_per_seg*3+2*n_obs_per_seg, (i+1)*n_obs_per_seg*3).tolist()
    x_train = x[ind_list_train,:,:]
    y_train = y[ind_list_train,:]
    s_train = s[ind_list_train,:,:]
    x_valid = x[ind_list_valid,:,:]
    y_valid = y[ind_list_valid,:]
    s_valid = s[ind_list_valid,:,:]
    x_test = x[ind_list_test,:,:]
    y_test = y[ind_list_test,:]
    s_test = s[ind_list_test,:,:]
    return x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test


def split_img_data(data):
    # with MGPCA
    # x_orig = data['x'] # subject x voxel x modality
    # w = data['w'] # feature x voxel x modality
    # x = np.concatenate([np.expand_dims(x_orig[:,:,0] @ w[:,:,0].T, axis=2), np.expand_dims(x_orig[:,:,1] @ w[:,:,1].T, axis=2)], axis=2)
    
    # without MGPCA
    x_train = data['x_train']
    x_valid = data['x_valid']
    x_test = data['x_test']
    # x_cat = np.concatenate([x_orig[:,:,0], x_orig[:,:,1]], axis=0)
    # pca = PCA(n_components=data_dim)
    # x_pca = pca.fit_transform(x_cat)
    # x = np.concatenate([np.expand_dims(x_pca[:x_orig.shape[0],:], axis=2), np.expand_dims(x_pca[x_orig.shape[0]:,:], axis=2)], axis=2) # 2907 x 30 x 2
    
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']

    u_train = to_one_hot(data['u_train'])[0]
    u_valid = to_one_hot(data['u_valid'])[0]
    u_test = to_one_hot(data['u_test'])[0]

    return x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test


def run_diva(args, config, method="diva"):
    # wandb.init(project=method, entity="deepmisa")
    start = time.time()
    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    index = slice(0, n_modality)
    experiment = config.experiment
    dataset = config.dataset
    n_layer = args.n_layer if args.n_layer else config.n_layer
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    hidden_dim = config.ivae.hidden_dim if config.ivae.hidden_dim != [] else latent_dim * 2
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = args.n_obs_per_seg if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch

    # MISA config
    input_dim = [latent_dim] * n_modality
    output_dim = [latent_dim] * n_modality
    subspace = config.subspace
    if subspace.lower() == 'iva':
        subspace = [torch.eye(dd, device=device) for dd in output_dim]

    eta = config.misa.eta
    beta = config.misa.beta
    lam = config.misa.lam

    if len(eta) > 0:
        eta = torch.tensor(eta, dtype=torch.float32, device=device)
        if len(eta) == 1:
            eta = eta*torch.ones(subspace[0].size(-2), device=device)
    
    if len(beta) > 0:
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        if len(beta) == 1:
            beta = beta*torch.ones(subspace[0].size(-2), device=device)
    
    if config.misa.fix_var:
        # set the variance to pi^2 / 3 for any parameter choice and any dimension
        nu = [(2*eta[i]+n_modality-1)/(2*beta[i]) for i in range(len(lam))]
        lam = [(3*math.gamma(nu[i]+(1/beta[i]))/((np.pi**2)*n_modality*math.gamma(nu[i])))**beta[i] for i in range(len(lam))]

    if len(lam) > 0:
        lam = torch.tensor(lam, dtype=torch.float32, device=device)
        if len(lam) == 1:
            lam = lam*torch.ones(subspace[0].size(-2), device=device)
    
    batch_size_misa = args.misa_batch_size if args.misa_batch_size else config.misa.batch_size
    init_method = config.init_method

    epoch_interval = 10 # save result every n epochs
    best_epoch = 0
    loss_ivae = {'train':np.zeros((n_epoch, n_modality)), 'valid':np.zeros((n_epoch, n_modality)), 'test':np.zeros((n_epoch, n_modality))}
    loss_misa = {'train':np.zeros((n_epoch, n_segment)), 'valid':np.zeros((n_epoch, n_segment)), 'test':np.zeros((n_epoch, n_segment))}
    res_mcc = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_metric = {e: {} for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_model_weight = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_recovered_source = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}

    if experiment == "sim":
        # generate synthetic data
        x, y, s = generate_synthetic_data(latent_dim, n_segment, n_obs_per_seg*3, n_layer, seed=data_seed,
            simulationMethod=dataset, one_hot_labels=True, varyMean=False)
        
        x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n_obs_per_seg)

        res_ground_truth_source = {'train': s_train, 'valid': s_valid, 'test': s_test}

        weight_init_list = []
    
    elif experiment == "img":
        data = sio.loadmat(data_path)
        
        x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

        res_ground_truth_source = {}

        weight_init_list = []
        # weight_init_list = [ np.eye(latent_dim) for _ in range(n_modality) ]
    
    lr_misa = lr_ivae/n_segment
    mi_misa = mi_ivae
    
    print(f'Running {method} experiment with L={n_layer}; n_obs_per_seg={n_obs_per_seg}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
    
    model_ivae_list = []
    ckpt_file_ivae_list = []
    data_dim = x_train.shape[1]
    aux_dim = y_train.shape[1]

    loader_params = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # create a list of iVAE modality-specific data loaders
    dset_list_train, dset_list_valid, dset_list_test = [], [], []
    mdl_list_train, mdl_list_valid, mdl_list_test = [], [], []
    
    for m in range(n_modality):
        dset_train = ConditionalDataset(x_train[:,:,m].astype(np.float32), y_train.astype(np.float32), device)
        data_loader_train = DataLoader(dset_train, shuffle=True, batch_size=batch_size_ivae, **loader_params)
        dset_valid = ConditionalDataset(x_valid[:,:,m].astype(np.float32), y_valid.astype(np.float32), device)
        data_loader_valid = DataLoader(dset_valid, shuffle=False, batch_size=batch_size_ivae, **loader_params)
        dset_test = ConditionalDataset(x_test[:,:,m].astype(np.float32), y_test.astype(np.float32), device)
        data_loader_test = DataLoader(dset_test, shuffle=False, batch_size=batch_size_ivae, **loader_params)
        
        dset_list_train.append(dset_train)
        dset_list_valid.append(dset_valid)
        dset_list_test.append(dset_test)
        mdl_list_train.append(data_loader_train)
        mdl_list_valid.append(data_loader_valid)
        mdl_list_test.append(data_loader_test)
        
    # create a list of MISA segment-specific data loaders
    np.random.seed(seed)
    segment_shuffled = np.arange(n_segment)
    np.random.shuffle(segment_shuffled)
    
    sdl_list_train, sdl_list_valid, sdl_list_test = [], [], []

    for seg in range(n_segment):
        if experiment == "sim":
            y_seg_train = y_train[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg]
            x_seg_train = x_train[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,:]
            y_seg_valid = y_valid[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg]
            x_seg_valid = x_valid[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,:]
            y_seg_test = y_test[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg]
            x_seg_test = x_test[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,:]
        elif experiment == "img":
            ind_train = np.where(u_train[:,seg]==1)[0]
            y_seg_train = y_train[ind_train,:]
            x_seg_train = x_train[ind_train,:,:]
            ind_valid = np.where(u_valid[:,seg]==1)[0]
            y_seg_valid = y_valid[ind_valid,:]
            x_seg_valid = x_valid[ind_valid,:,:]
            ind_test = np.where(u_test[:,seg]==1)[0]
            y_seg_test = y_test[ind_test,:]
            x_seg_test = x_test[ind_test,:,:]
        
        # remove mean of segment
        x_seg_dm_train = x_seg_train - np.mean(x_seg_train, axis=0)
        x_seg_dm_valid = x_seg_valid - np.mean(x_seg_valid, axis=0)
        x_seg_dm_test = x_seg_test - np.mean(x_seg_test, axis=0)
        ds_train = ConditionalDataset(x_seg_dm_train.astype(np.float32), y_seg_train.astype(np.float32), device)
        ds_valid = ConditionalDataset(x_seg_dm_valid.astype(np.float32), y_seg_valid.astype(np.float32), device)
        ds_test = ConditionalDataset(x_seg_dm_test.astype(np.float32), y_seg_test.astype(np.float32), device)
        data_loader_train = DataLoader(ds_train, shuffle=False, batch_size=len(ds_train), **loader_params)
        data_loader_valid = DataLoader(ds_valid, shuffle=False, batch_size=len(ds_valid), **loader_params)
        data_loader_test = DataLoader(ds_test, shuffle=False, batch_size=len(ds_test), **loader_params)
        sdl_list_train.append(data_loader_train)
        sdl_list_valid.append(data_loader_valid)
        sdl_list_test.append(data_loader_test)

    # initiate iVAE model for each modality
    for m in range(n_modality):
        ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_ivae_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}_modality{m+1}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
        ckpt_file_ivae_list.append(ckpt_file)
        
        model_ivae = iVAE(latent_dim, 
                        data_dim, 
                        aux_dim, 
                        activation='lrelu', 
                        device=device, 
                        n_layer=n_layer, 
                        hidden_dim=hidden_dim,
                        method=method,
                        seed=seed)
        
        if init_method == 'mvica':
            # initialize iVAE weights with multi-view ICA initial sources from PCA and perm ICA
            fname = os.path.join(args.run, f"res_mvica_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}.p")
            if os.path.exists(fname):
                print(f'Loading multi-view ICA weight {fname}')
                res = pickle.load(open(fname, 'rb'))
                s_init_mvica = res['initial_recovered_source_per_modality'][m]['train']
                model_ivae = IVAE_init_wrapper(X=x_train[:,:,m], U=y_train, S=s_init_mvica, model=model_ivae, batch_size=batch_size_ivae, mi_ivae=1000)
            else:
                print(f'File {fname} not found. Initialize iVAE using random weights.')

        model_ivae, _ = IVAE_wrapper(data_loader=mdl_list_train[m], batch_size=batch_size_ivae, 
                                    n_layer=n_layer, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae,
                                    ckpt_file=ckpt_file, seed=seed, model=model_ivae, 
                                    data_loader_valid=mdl_list_valid[m])
        
        X_train, U_train = dset_list_train[m].x, dset_list_train[m].y
        _, _, rs_ivae_train, _ = model_ivae(X_train, U_train)
        X_valid, U_valid = dset_list_valid[m].x, dset_list_valid[m].y
        _, _, rs_ivae_valid, _ = model_ivae(X_valid, U_valid)

        rs_ivae_train = rs_ivae_train.detach().numpy()
        fname = os.path.join(args.run, f'src_ivae_m{m+1}_{experiment}_{method}_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
        pickle.dump(rs_ivae_train, open(fname, "wb"))

        model_ivae_list.append(model_ivae)
    
    for m in range(n_modality):
        model_ivae_list[m].set_aux(False)

    model_misa = MISA(weights=weight_init_list,
                    index=index, 
                    subspace=subspace, 
                    eta=eta, 
                    beta=beta, 
                    lam=lam, 
                    input_dim=input_dim, 
                    output_dim=output_dim, 
                    seed=seed, 
                    device=device,
                    model=model_ivae_list,
                    latent_dim=latent_dim)

    ckpt_file_misa = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_misa_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}_epoch{n_epoch}_maxiter{mi_ivae}_lrmisa{lr_misa}.pt')

    # update iVAE and MISA model weights
    # run iVAE per modality
    for e in range(n_epoch):
        print(f'\nEpoch: {e}')
        # loop MISA through segments
        # remove the mean of segment because MISA loss assumes zero mean
        # randomize segment order
        for it in range(mi_misa):
            for seg in segment_shuffled:
                model_misa, [rs_misa_train, rs_misa_valid], [loss_misa['train'][e,seg], loss_misa['valid'][e,seg]], res_misa_dict = \
                    MISA_wrapper_(data_loader=sdl_list_train[seg],
                                test_data_loader=sdl_list_valid[seg],
                                epochs=1,
                                lr=lr_misa,
                                device=device,
                                ckpt_file=ckpt_file_misa,
                                model_MISA=model_misa)
                
                _, rs_misa_test, loss_misa['test'][e,seg], _ = \
                    MISA_wrapper_(test=True, data_loader=sdl_list_test[seg], device=device, 
                                ckpt_file=ckpt_file_misa, model_MISA=model_misa)
        
        if e % epoch_interval == 0:
            for m in range(n_modality):
                for l in range(n_layer):
                    res_model_weight[e].append(model_misa.input_model[m].g.fc[l].weight)

        for m in range(n_modality):
            model_misa.input_model[m].set_aux(True)
            # print(f"model_misa.input_model[{m}].use_aux = {model_misa.input_model[m].use_aux}")
            model_misa.input_model[m], [loss_ivae['train'][e,m], loss_ivae['valid'][e,m]] = \
                IVAE_wrapper(data_loader=mdl_list_train[m], batch_size=batch_size_ivae, 
                            n_layer=n_layer, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                            ckpt_file=ckpt_file_ivae_list[m], seed=seed, test=False, 
                            model=model_misa.input_model[m], data_loader_valid=mdl_list_valid[m])
            
            _, loss_ivae['test'][e,m] = \
                IVAE_wrapper(data_loader=mdl_list_test[m], batch_size=batch_size_ivae, 
                            n_layer=n_layer, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                            ckpt_file=ckpt_file_ivae_list[m], seed=seed, test=True, 
                            model=model_misa.input_model[m])
            
            X_train, U_train = dset_list_train[m].x, dset_list_train[m].y
            _, _, rs_ivae_train, _ = model_misa.input_model[m](X_train, U_train)
            X_valid, U_valid = dset_list_valid[m].x, dset_list_valid[m].y
            _, _, rs_ivae_valid, _ = model_misa.input_model[m](X_valid, U_valid)
            X_test, U_test = dset_list_test[m].x, dset_list_test[m].y
            _, _, rs_ivae_test, _ = model_misa.input_model[m](X_test, U_test)
            
            model_misa.input_model[m].set_aux(False)
            # print(f"model_misa.input_model[{m}].use_aux = {model_misa.input_model[m].use_aux}")

            # store test results every epoch_interval epochs
            if e % epoch_interval == 0:
                rs_ivae_train = rs_ivae_train.detach().numpy()
                rs_ivae_valid = rs_ivae_valid.detach().numpy()
                rs_ivae_test = rs_ivae_test.detach().numpy()
                res_recovered_source[e].append({'train':rs_ivae_train, 'valid':rs_ivae_valid, 'test':rs_ivae_test})
                
                if experiment == 'sim':
                    res_mcc[e].append({'train': mean_corr_coef_per_segment(rs_ivae_train, s_train[:,:,m], y_train), \
                                        'valid': mean_corr_coef_per_segment(rs_ivae_valid, s_valid[:,:,m], y_valid), \
                                        'test': mean_corr_coef_per_segment(rs_ivae_test, s_test[:,:,m], y_test)})
                    if m == n_modality - 1:
                        metric_train = MMSE(np.dstack([r['train'] for r in res_recovered_source[e]]), s_train, y_train)
                        metric_valid = MMSE(np.dstack([r['valid'] for r in res_recovered_source[e]]), s_valid, y_valid)
                        metric_test = MMSE(np.dstack([r['test'] for r in res_recovered_source[e]]), s_test, y_test)
                        res_metric[e] = {'train': metric_train, 'valid': metric_valid, 'test': metric_test}

                        current_mcc_valid = np.mean([res_mcc[e][m]['valid'][0][0] for m in range(n_modality)])
                        if e == 0:
                            best_mcc_valid = current_mcc_valid
                        if current_mcc_valid >= best_mcc_valid:
                            for m in range(n_modality):
                                ckpt_file_ivae_best = ckpt_file_ivae_list[m].replace('.pt', '_best.pt')
                                shutil.copyfile(ckpt_file_ivae_list[m], ckpt_file_ivae_best)
                            ckpt_file_misa_best = ckpt_file_misa.replace('.pt', '_best.pt')
                            shutil.copyfile(ckpt_file_misa, ckpt_file_misa_best)
                            loss_ivae_valid = np.mean(loss_ivae['valid'][e,:])
                            loss_misa_valid = np.mean(loss_misa['valid'][e,:]) # average across segments
                            print(f"Epoch: {e}; IVAE Validation loss: {loss_ivae_valid}; MISA Validation loss: {loss_misa_valid}; Validation MCC: {current_mcc_valid}; Saved checkpoint to: {ckpt_file_misa_best}")
                            best_mcc_valid = current_mcc_valid
                            best_epoch = e
    
    end = time.time()
    t = (end - start) / 60
    print(f"Total time: {t} minutes")
    
    # prepare output
    Results = {
        'loss_ivae': loss_ivae,
        'loss_misa': loss_misa,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'mcc': res_mcc,
        'metric': res_metric,
        'model_weight': res_model_weight,
        'best_epoch': best_epoch
    }

    return Results


def run_ivae(args, config, method="ivae"):
    # wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    dataset = config.dataset
    n_layer = args.n_layer if args.n_layer else config.n_layer
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    hidden_dim = config.ivae.hidden_dim if config.ivae.hidden_dim != [] else latent_dim * 2
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = args.n_obs_per_seg if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch
    init_method = config.init_method
    
    epoch_interval = 10 # save result every n epochs
    best_epoch = np.zeros(n_modality)
    loss = {'train':np.zeros((n_epoch, n_modality)), 'valid':np.zeros((n_epoch, n_modality)), 'test':np.zeros((n_epoch, n_modality))}
    res_mcc = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_metric = {e: {} for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_recovered_source = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    
    if experiment == "sim":
        # generate synthetic data
        x, y, s = generate_synthetic_data(latent_dim, n_segment, n_obs_per_seg*3, n_layer, seed=data_seed,
                                        simulationMethod=dataset, one_hot_labels=True, varyMean=False)
        
        x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n_obs_per_seg)

        res_ground_truth_source = {'train': s_train, 'valid': s_valid, 'test': s_test}

    elif experiment == "img":
        data = sio.loadmat(data_path)
        
        x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test  = split_img_data(data)

        res_ground_truth_source = {}

    print(f'Running {method} experiment with L={n_layer}; n_obs_per_seg={n_obs_per_seg}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
    
    # intiate iVAE model for each modality
    model_ivae_list = []
    data_dim = x_train.shape[1]
    aux_dim = y_train.shape[1]

    loader_params = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # create a list of iVAE modality-specific data loaders
    dset_list_train, dset_list_valid, dset_list_test = [], [], []
    mdl_list_train, mdl_list_valid, mdl_list_test = [], [], []
    
    for m in range(n_modality):
        dset_train = ConditionalDataset(x_train[:,:,m].astype(np.float32), y_train.astype(np.float32), device)
        data_loader_train = DataLoader(dset_train, shuffle=True, batch_size=batch_size_ivae, **loader_params)
        dset_valid = ConditionalDataset(x_valid[:,:,m].astype(np.float32), y_valid.astype(np.float32), device)
        data_loader_valid = DataLoader(dset_valid, shuffle=False, batch_size=batch_size_ivae, **loader_params)
        dset_test = ConditionalDataset(x_test[:,:,m].astype(np.float32), y_test.astype(np.float32), device)
        data_loader_test = DataLoader(dset_test, shuffle=False, batch_size=batch_size_ivae, **loader_params)
        
        dset_list_train.append(dset_train)
        dset_list_valid.append(dset_valid)
        dset_list_test.append(dset_test)
        mdl_list_train.append(data_loader_train)
        mdl_list_valid.append(data_loader_valid)
        mdl_list_test.append(data_loader_test)

    for m in range(n_modality):
        ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}_modality{m+1}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
        
        model_ivae = iVAE(latent_dim, 
                        data_dim, 
                        aux_dim, 
                        activation='lrelu', 
                        device=device, 
                        n_layer=n_layer, 
                        hidden_dim=hidden_dim,
                        method=method,
                        seed=seed)
        
        if init_method == 'mvica':
            # initialize iVAE weights with multi-view ICA initial sources from PCA and perm ICA
            fname = os.path.join(args.run, f"res_mvica_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}.p")
            if os.path.exists(fname):
                print(f'Loading multi-view ICA weight {fname}')
                res = pickle.load(open(fname, 'rb'))
                s_init_mvica = res['initial_recovered_source_per_modality'][m]['train']
                model_ivae = IVAE_init_wrapper(X=x_train[:,:,m], U=y_train, S=s_init_mvica, model=model_ivae, batch_size=batch_size_ivae)
            else:
                print(f'File {fname} not found. Initialize iVAE using random weights.')

        for e in range(n_epoch):
            print(f'Epoch: {e}')
            model_ivae, [loss['train'][e,m], loss['valid'][e,m]] = IVAE_wrapper(data_loader=mdl_list_train[m], batch_size=batch_size_ivae, 
                                                                                n_layer=n_layer, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                                                                                ckpt_file=ckpt_file, seed=seed, test=False, 
                                                                                model=model_ivae, data_loader_valid=mdl_list_valid[m])
            
            _, loss['test'][e,m] = IVAE_wrapper(data_loader=mdl_list_test[m], batch_size=batch_size_ivae, 
                                                n_layer=n_layer, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                                                ckpt_file=ckpt_file, seed=seed, test=True, model=model_ivae)
            
            if e % epoch_interval == 0:
                X_train, U_train = dset_list_train[m].x, dset_list_train[m].y
                _, _, rs_ivae_train, _ = model_ivae(X_train, U_train)
                X_valid, U_valid = dset_list_valid[m].x, dset_list_valid[m].y
                _, _, rs_ivae_valid, _ = model_ivae(X_valid, U_valid)
                X_test, U_test = dset_list_test[m].x, dset_list_test[m].y
                _, _, rs_ivae_test, _ = model_ivae(X_test, U_test)

                rs_ivae_train = rs_ivae_train.detach().numpy()
                rs_ivae_valid = rs_ivae_valid.detach().numpy()
                rs_ivae_test = rs_ivae_test.detach().numpy()
                res_recovered_source[e].append({'train': rs_ivae_train, 'valid': rs_ivae_valid, 'test': rs_ivae_test})

                if experiment == 'sim':
                    res_mcc[e].append({'train': mean_corr_coef_per_segment(rs_ivae_train, s_train[:,:,m], y_train), \
                                        'valid': mean_corr_coef_per_segment(rs_ivae_valid, s_valid[:,:,m], y_valid), \
                                        'test': mean_corr_coef_per_segment(rs_ivae_test, s_test[:,:,m], y_test)})
                    
                    current_mcc_valid = res_mcc[e][m]['valid'][0][0]
                    if e == 0:
                        best_mcc_valid = current_mcc_valid
                    if current_mcc_valid >= best_mcc_valid:
                        ckpt_file_best = ckpt_file.replace('.pt', '_best.pt')
                        shutil.copyfile(ckpt_file, ckpt_file_best)
                        loss_valid = loss['valid'][e,m]
                        print(f"Epoch: {e}; Validation loss: {loss_valid}; Validation MCC: {current_mcc_valid}; Saved checkpoint to: {ckpt_file_best}")
                        best_mcc_valid = current_mcc_valid
                        best_epoch[m] = e
                    
                    if m == n_modality - 1:
                        metric_train = MMSE(np.dstack([r['train'] for r in res_recovered_source[e]]), s_train, y_train)
                        metric_valid = MMSE(np.dstack([r['valid'] for r in res_recovered_source[e]]), s_valid, y_valid)
                        metric_test = MMSE(np.dstack([r['test'] for r in res_recovered_source[e]]), s_test, y_test)
                        res_metric[e] = {'train': metric_train, 'valid': metric_valid, 'test': metric_test}

        model_ivae_list.append(model_ivae)

    # prepare output
    Results = {
        'loss': loss,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'mcc': res_mcc,
        'metric': res_metric,
        'best_epoch': best_epoch
    }

    return Results


def run_jivae(args, config, method="jivae"):
    # wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    dataset = config.dataset
    n_layer = args.n_layer if args.n_layer else config.n_layer
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    hidden_dim = config.ivae.hidden_dim if config.ivae.hidden_dim != [] else latent_dim * 2
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = args.n_obs_per_seg if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch
    init_method = config.init_method

    epoch_interval = 10 # save result every n epochs
    best_epoch = 0
    loss = {'train':np.zeros(n_epoch), 'valid':np.zeros(n_epoch), 'test':np.zeros(n_epoch)}
    res_mcc = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_metric = {e: {} for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_recovered_source = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    
    if experiment == "sim":
        # generate synthetic data
        x, y, s = generate_synthetic_data(latent_dim, n_segment, n_obs_per_seg*3, n_layer, seed=data_seed,
                                        simulationMethod=dataset, one_hot_labels=True, varyMean=False)
        
        x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n_obs_per_seg)

        res_ground_truth_source = {'train': s_train, 'valid': s_valid, 'test': s_test}

    elif experiment == "img":
        data = sio.loadmat(data_path)
        
        x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

        res_ground_truth_source = {}

    print(f'Running {method} experiment with L={n_layer}; n_obs_per_seg={n_obs_per_seg}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
    
    # run a single iVAE model on concatenated modalities along sample dimension
    ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
    x_concat_train = np.vstack([x_train[:,:,m] for m in range(n_modality)])
    y_concat_train = np.concatenate([y_train]*n_modality, axis=0)
    x_concat_valid = np.vstack([x_valid[:,:,m] for m in range(n_modality)])
    y_concat_valid = np.concatenate([y_valid]*n_modality, axis=0)
    x_concat_test = np.vstack([x_test[:,:,m] for m in range(n_modality)])
    y_concat_test = np.concatenate([y_test]*n_modality, axis=0)

    data_dim = x_concat_train.shape[1]
    aux_dim = y_concat_train.shape[1]
    n_sample_train = x_train.shape[0]
    n_sample_valid = x_valid.shape[0]
    n_sample_test = x_test.shape[0]

    loader_params = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # create iVAE data loaders
    dset_train = ConditionalDataset(x_concat_train.astype(np.float32), y_concat_train.astype(np.float32), device)
    data_loader_train = DataLoader(dset_train, shuffle=True, batch_size=batch_size_ivae, **loader_params)
    dset_valid = ConditionalDataset(x_concat_valid.astype(np.float32), y_concat_valid.astype(np.float32), device)
    data_loader_valid = DataLoader(dset_valid, shuffle=False, batch_size=batch_size_ivae, **loader_params)
    dset_test = ConditionalDataset(x_concat_test.astype(np.float32), y_concat_test.astype(np.float32), device)
    data_loader_test = DataLoader(dset_test, shuffle=False, batch_size=batch_size_ivae, **loader_params)
    
    model_ivae = iVAE(latent_dim, 
                    data_dim, 
                    aux_dim, 
                    activation='lrelu', 
                    device=device, 
                    n_layer=n_layer, 
                    hidden_dim=hidden_dim,
                    method=method,
                    seed=seed)
    
    if init_method == 'mvica':
        # initialize iVAE weights with multi-view ICA initial sources from PCA and perm ICA
        fname = os.path.join(args.run, f"res_mvica_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}.p")
        if os.path.exists(fname):
            print(f'Loading multi-view ICA weight {fname}')
            res = pickle.load(open(fname, 'rb'))
            s_init_mvica = np.vstack([res['initial_recovered_source_per_modality'][m]['train'] for m in range(n_modality)])
            model_ivae = IVAE_init_wrapper(X=x_concat_train, U=y_concat_train, S=s_init_mvica, model=model_ivae, batch_size=batch_size_ivae)
        else:
            print(f'File {fname} not found. Initialize iVAE using random weights.')

    for e in range(n_epoch):
        print(f'Epoch: {e}')
        model_ivae, [loss['train'][e], loss['valid'][e]] = \
            IVAE_wrapper(data_loader=data_loader_train, batch_size=batch_size_ivae, 
                        n_layer=n_layer, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                        ckpt_file=ckpt_file, seed=seed, test=False, model=model_ivae, 
                        data_loader_valid=data_loader_valid)

        _, loss['test'][e] = \
            IVAE_wrapper(data_loader=data_loader_test, batch_size=batch_size_ivae, n_layer=n_layer, cuda=cuda, 
                        max_iter=mi_ivae, lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=True, model=model_ivae)

        if e % epoch_interval == 0:
            X_train, U_train = dset_train.x, dset_train.y
            _, _, rs_ivae_train, _ = model_ivae(X_train, U_train)
            X_valid, U_valid = dset_valid.x, dset_valid.y
            _, _, rs_ivae_valid, _ = model_ivae(X_valid, U_valid)
            X_test, U_test = dset_test.x, dset_test.y
            _, _, rs_ivae_test, _ = model_ivae(X_test, U_test)

            rs_ivae_train = rs_ivae_train.detach().numpy()
            rs_ivae_valid = rs_ivae_valid.detach().numpy()
            rs_ivae_test = rs_ivae_test.detach().numpy()

            for m in range(n_modality):
                res_recovered_source[e].append({'train':rs_ivae_train[m*n_sample_train:(m+1)*n_sample_train,:], \
                                                'valid':rs_ivae_valid[m*n_sample_valid:(m+1)*n_sample_valid,:], \
                                                'test':rs_ivae_test[m*n_sample_test:(m+1)*n_sample_test,:]})
                
            if experiment == 'sim':
                for m in range(n_modality):
                    res_mcc[e].append({'train': mean_corr_coef_per_segment(rs_ivae_train[m*n_sample_train:(m+1)*n_sample_train,:], s_train[:,:,m], y_train), \
                                       'valid': mean_corr_coef_per_segment(rs_ivae_valid[m*n_sample_valid:(m+1)*n_sample_valid,:], s_valid[:,:,m], y_valid), \
                                       'test': mean_corr_coef_per_segment(rs_ivae_test[m*n_sample_test:(m+1)*n_sample_test,:], s_test[:,:,m], y_test)})
                
                metric_train = MMSE(np.dstack([r['train'] for r in res_recovered_source[e]]), s_train, y_train)
                metric_valid = MMSE(np.dstack([r['valid'] for r in res_recovered_source[e]]), s_valid, y_valid)
                metric_test = MMSE(np.dstack([r['test'] for r in res_recovered_source[e]]), s_test, y_test)
                res_metric[e] = {'train': metric_train, 'valid': metric_valid, 'test': metric_test}

                current_mcc_valid = np.mean([res_mcc[e][m]['valid'][0][0] for m in range(n_modality)])
                if e == 0:
                    best_mcc_valid = current_mcc_valid
                if current_mcc_valid >= best_mcc_valid:
                    ckpt_file_best = ckpt_file.replace('.pt', '_best.pt')
                    shutil.copyfile(ckpt_file, ckpt_file_best)
                    loss_valid = loss['valid'][e]
                    print(f"Epoch: {e}; Validation loss: {loss_valid}; Validation MCC: {current_mcc_valid}; Saved checkpoint to: {ckpt_file_best}")
                    best_mcc_valid = current_mcc_valid
                    best_epoch = e
    
    # prepare output
    Results = {
        'loss': loss,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'mcc': res_mcc,
        'metric': res_metric,
        'best_epoch': best_epoch
    }

    return Results


def run_givae(args, config, method="givae"):
    # wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    dataset = config.dataset
    n_layer = args.n_layer if args.n_layer else config.n_layer
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    hidden_dim = config.ivae.hidden_dim if config.ivae.hidden_dim != [] else latent_dim * 2
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = args.n_obs_per_seg if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch
    init_method = config.init_method

    epoch_interval = 10 # save result every n epochs
    best_epoch = 0
    loss = {'train':np.zeros(n_epoch), 'valid':np.zeros(n_epoch), 'test':np.zeros(n_epoch)}
    res_mcc = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_metric = {e: {} for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_recovered_source = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}

    if experiment == "sim":
        # generate synthetic data
        x, y, s = generate_synthetic_data(latent_dim, n_segment, n_obs_per_seg*3, n_layer, seed=data_seed,
            simulationMethod=dataset, one_hot_labels=True, varyMean=False)
        
        x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n_obs_per_seg)

        res_ground_truth_source = {'train': s_train, 'valid': s_valid, 'test': s_test}

    elif experiment == "img":
        data = sio.loadmat(data_path)
        
        x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

        res_ground_truth_source = {}

    print(f'Running {method} experiment with L={n_layer}; n_obs_per_seg={n_obs_per_seg}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
    
    # run a single iVAE model on concatenated modalities along feature dimension
    ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_givae_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
    x_concat_train = np.hstack([x_train[:,:,m] for m in range(n_modality)])
    x_concat_valid = np.hstack([x_valid[:,:,m] for m in range(n_modality)])
    x_concat_test = np.hstack([x_test[:,:,m] for m in range(n_modality)])
    data_dim = x_concat_train.shape[1]
    aux_dim = y_train.shape[1]

    loader_params = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # create iVAE data loaders
    dset_train = ConditionalDataset(x_concat_train.astype(np.float32), y_train.astype(np.float32), device)
    data_loader_train = DataLoader(dset_train, shuffle=True, batch_size=batch_size_ivae, **loader_params)
    dset_valid = ConditionalDataset(x_concat_valid.astype(np.float32), y_valid.astype(np.float32), device)
    data_loader_valid = DataLoader(dset_valid, shuffle=False, batch_size=batch_size_ivae, **loader_params)
    dset_test = ConditionalDataset(x_concat_test.astype(np.float32), y_test.astype(np.float32), device)
    data_loader_test = DataLoader(dset_test, shuffle=False, batch_size=batch_size_ivae, **loader_params)

    model_ivae = iVAE(latent_dim, 
                    data_dim, 
                    aux_dim, 
                    activation='lrelu', 
                    device=device, 
                    n_layer=n_layer, 
                    hidden_dim=hidden_dim,
                    method=method,
                    seed=seed)
    
    if init_method == 'mvica':
        # initialize iVAE weights with multi-view ICA initial sources from PCA and perm ICA
        fname = os.path.join(args.run, f"res_mvica_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}.p")
        if os.path.exists(fname):
            print(f'Loading multi-view ICA weight {fname}')
            res = pickle.load(open(fname, 'rb'))
            s_init_mvica = np.mean(np.dstack([res['initial_recovered_source_per_modality'][m]['train'] for m in range(n_modality)]), axis=2)
            model_ivae = IVAE_init_wrapper(X=x_concat_train, U=y_train, S=s_init_mvica, model=model_ivae, batch_size=batch_size_ivae)
        else:
            print(f'File {fname} not found. Initialize iVAE using random weights.')

    for e in range(n_epoch):
        print(f'Epoch: {e}')
        model_ivae, [loss['train'][e], loss['valid'][e]] = \
            IVAE_wrapper(data_loader=data_loader_train, batch_size=batch_size_ivae, n_layer=n_layer, 
                        cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, 
                        test=False, model=model_ivae, data_loader_valid=data_loader_valid)
        
        _, loss['test'][e] = \
            IVAE_wrapper(data_loader=data_loader_test, batch_size=batch_size_ivae, n_layer=n_layer, cuda=cuda, 
                        max_iter=mi_ivae, lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=True, model=model_ivae)
        
        if e % epoch_interval == 0:
            X_train, U_train = dset_train.x, dset_train.y
            _, _, rs_ivae_train, _ = model_ivae(X_train, U_train)
            X_valid, U_valid = dset_valid.x, dset_valid.y
            _, _, rs_ivae_valid, _ = model_ivae(X_valid, U_valid)
            X_test, U_test = dset_test.x, dset_test.y
            _, _, rs_ivae_test, _ = model_ivae(X_test, U_test)
            
            rs_ivae_test = rs_ivae_test.detach().numpy()
            rs_ivae_train = rs_ivae_train.detach().numpy()
            rs_ivae_valid = rs_ivae_valid.detach().numpy()
            for m in range(n_modality):
                res_recovered_source[e].append({'test':rs_ivae_test, 'train':rs_ivae_train, 'valid':rs_ivae_valid})

            if experiment == 'sim':
                for m in range(n_modality):
                    res_mcc[e].append({'train': mean_corr_coef_per_segment(rs_ivae_train, s_train[:,:,m], y_train), \
                                        'valid': mean_corr_coef_per_segment(rs_ivae_valid, s_valid[:,:,m], y_valid), \
                                        'test': mean_corr_coef_per_segment(rs_ivae_test, s_test[:,:,m], y_test)})
                
                metric_train = MMSE(np.dstack([r['train'] for r in res_recovered_source[e]]), s_train, y_train)
                metric_valid = MMSE(np.dstack([r['valid'] for r in res_recovered_source[e]]), s_valid, y_valid)
                metric_test = MMSE(np.dstack([r['test'] for r in res_recovered_source[e]]), s_test, y_test)
                res_metric[e] = {'train': metric_train, 'valid': metric_valid, 'test': metric_test}

                current_mcc_valid = np.mean([res_mcc[e][m]['valid'][0][0] for m in range(n_modality)])
                if e == 0:
                    best_mcc_valid = current_mcc_valid
                if current_mcc_valid >= best_mcc_valid:
                    ckpt_file_best = ckpt_file.replace('.pt', '_best.pt')
                    shutil.copyfile(ckpt_file, ckpt_file_best)
                    loss_valid = loss['valid'][e]
                    print(f"Epoch: {e}; Validation loss: {loss_valid}; Validation MCC: {current_mcc_valid}; Saved checkpoint to: {ckpt_file_best}")
                    best_mcc_valid = current_mcc_valid
                    best_epoch = e

    # prepare output
    Results = {
        'loss': loss,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'mcc': res_mcc,
        'metric': res_metric,
        'best_epoch': best_epoch
    }

    return Results


def run_deepmisa(args, config, method="misa"):
    # wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    index = slice(0, n_modality)
    experiment = config.experiment
    dataset = config.dataset
    n_layer = args.n_layer if args.n_layer else config.n_layer
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = args.n_obs_per_seg if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch

    # MISA config
    input_dim = [latent_dim] * n_modality
    output_dim = [latent_dim] * n_modality
    subspace = config.subspace
    if subspace.lower() == 'iva':
        subspace = [torch.eye(dd, device=device) for dd in output_dim]

    eta = config.misa.eta
    beta = config.misa.beta
    lam = config.misa.lam
    if len(eta) > 0:
        eta = torch.tensor(eta, dtype=torch.float32, device=device)
        if len(eta) == 1:
            eta = eta*torch.ones(subspace[0].size(-2), device=device)
    
    if len(beta) > 0:
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        if len(beta) == 1:
            beta = beta*torch.ones(subspace[0].size(-2), device=device)
    
    if len(lam) > 0:
        lam = torch.tensor(lam, dtype=torch.float32, device=device)
        if len(lam) == 1:
            lam = lam*torch.ones(subspace[0].size(-2), device=device)
    
    batch_size_misa = args.misa_batch_size if args.misa_batch_size else config.misa.batch_size
    init_method = config.init_method
    
    epoch_interval = 10 # save result every n epochs
    best_epoch = 0
    loss = {'train':np.zeros((n_epoch, n_segment)), 'valid':np.zeros((n_epoch, n_segment)), 'test':np.zeros((n_epoch, n_segment))}
    res_mcc = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_metric = {e: {} for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_recovered_source = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}

    if experiment == "sim":
        # generate synthetic data
        x, y, s = generate_synthetic_data(latent_dim, n_segment, n_obs_per_seg*3, n_layer, seed=data_seed,
            simulationMethod=dataset, one_hot_labels=True, varyMean=False)
        
        x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n_obs_per_seg)

        res_ground_truth_source = {'train': s_train, 'valid': s_valid, 'test': s_test}

        weight_init_list = []
    
    elif experiment == "img":
        data = sio.loadmat(data_path)
        
        x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

        res_ground_truth_source = {}

        weight_init_list = []
        # weight_init_list = [ np.eye(data_dim) for _ in range(n_modality) ]
    
    lr_misa = lr_ivae/n_segment
    mi_misa = mi_ivae
    
    print(f'Running {method} experiment with L={n_layer}; n_obs_per_seg={n_obs_per_seg}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')

    ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}_epoch{n_epoch}_maxiter{mi_misa}_lrmisa{round(lr_misa, 5)}.pt')
    
    # weight initialization
    for m in range(n_modality):
        # initialize MISA model weights using iVAE sources as A = (z^T z)^{-1} z^T X
        if init_method == 'ivae':
            fname = os.path.join(args.run, f'src_ivae_m{m+1}_{experiment}_diva_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
            if os.path.exists(fname):
                print(f'Loading iVAE source {fname}')
                s_ivae = pickle.load(open(fname, 'rb'))
                weight_init = np.linalg.inv(s_ivae.T @ s_ivae) @ s_ivae.T @ x_train[:,:,m]
                weight_init_list.append(weight_init)
            else:
                print(f'File {fname} not found. Initialize MISA using random weights.')
        # initialize MISA model weights using multi-view ICA final weights
        elif init_method == 'mvica':
            fname = os.path.join(args.run, f"res_mvica_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}.p")
            if os.path.exists(fname):
                print(f'Loading multi-view ICA weight {fname}')
                res = pickle.load(open(fname, 'rb'))
                weight_init = res['weight_init']['train'][m,:,:]
                weight_init_list.append(weight_init)
            else:
                print(f'File {fname} not found. Initialize MISA using random weights.')
    
    model_misa = MISA(weights=weight_init_list,
        index=index, 
        subspace=subspace, 
        eta=eta, 
        beta=beta, 
        lam=lam, 
        input_dim=input_dim, 
        output_dim=output_dim, 
        seed=seed, 
        device=device,
        latent_dim=latent_dim)

    # update iVAE and MISA model weights
    # run iVAE per modality
    np.random.seed(seed)
    segment_shuffled = np.arange(n_segment)
    np.random.shuffle(segment_shuffled)

    loader_params = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    sdl_list_train, sdl_list_valid, sdl_list_test = [], [], []

    for seg in range(n_segment):
        if experiment == "sim":
            x_seg_train = x_train[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,:]
            x_seg_valid = x_valid[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,:]
            x_seg_test = x_test[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,:]
        elif experiment == "img":
            ind_train = np.where(u_train[:,seg]==1)[0]
            x_seg_train = x_train[ind_train,:,:]
            ind_valid = np.where(u_valid[:,seg]==1)[0]
            x_seg_valid = x_valid[ind_valid,:,:]
            ind_test = np.where(u_test[:,seg]==1)[0]
            x_seg_test = x_test[ind_test,:,:]
        
        # remove mean of segment
        x_seg_dm_train = x_seg_train - np.mean(x_seg_train, axis=0)
        x_seg_dm_valid = x_seg_valid - np.mean(x_seg_valid, axis=0)
        x_seg_dm_test = x_seg_test - np.mean(x_seg_test, axis=0)

        # a list of datasets, each dataset dimension is sample x source
        ds_train = Dataset(data_in=x_seg_dm_train, device=device)
        ds_valid = Dataset(data_in=x_seg_dm_valid, device=device)
        ds_test = Dataset(data_in=x_seg_dm_test, device=device)
        data_loader_train = DataLoader(dataset=ds_train, batch_size=len(ds_train), shuffle=False, **loader_params)
        data_loader_valid = DataLoader(dataset=ds_valid, batch_size=len(ds_valid), shuffle=False, **loader_params)
        data_loader_test = DataLoader(dataset=ds_test, batch_size=len(ds_test), shuffle=False, **loader_params)

        sdl_list_train.append(data_loader_train)
        sdl_list_valid.append(data_loader_valid)
        sdl_list_test.append(data_loader_test)

    if experiment == "sim":
        res_rs_misa_train = np.zeros_like(s_train)
        res_rs_misa_valid = np.zeros_like(s_valid)
        res_rs_misa_test = np.zeros_like(s_test)
    else:
        res_rs_misa_train = np.zeros((x_train.shape[0], latent_dim, n_modality))
        res_rs_misa_valid = np.zeros((x_valid.shape[0], latent_dim, n_modality))
        res_rs_misa_test = np.zeros((x_test.shape[0], latent_dim, n_modality))

    for e in range(n_epoch):
        print(f'Epoch: {e}')
        # loop MISA through segments
        # remove the mean of segment because MISA loss assumes zero mean
        # randomize segment order
        for seg in segment_shuffled:
            model_misa, [rs_misa_train, rs_misa_valid], [loss['train'][e,seg], loss['valid'][e,seg]], res_misa_dict = \
                MISA_wrapper_(data_loader=sdl_list_train[seg],
                              test_data_loader=sdl_list_valid[seg],
                              epochs=mi_misa, lr=lr_misa, device=device,
                              ckpt_file=ckpt_file, model_MISA=model_misa)

            _, rs_misa_test, loss['test'][e,seg], _ = \
                MISA_wrapper_(test=True, data_loader=sdl_list_test[seg], device=device, 
                              ckpt_file=ckpt_file, model_MISA=model_misa)
            
            for m in range(n_modality):
                if experiment == "sim":
                    res_rs_misa_train[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,m] = rs_misa_train[m].detach().numpy()
                    res_rs_misa_valid[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,m] = rs_misa_valid[m].detach().numpy()
                    res_rs_misa_test[seg*n_obs_per_seg:(seg+1)*n_obs_per_seg,:,m] = rs_misa_test[m].detach().numpy()
                elif experiment == "img":
                    ind_train = np.where(u_train[:,seg]==1)[0]
                    ind_valid = np.where(u_valid[:,seg]==1)[0]
                    ind_test = np.where(u_test[:,seg]==1)[0]
                    res_rs_misa_train[ind_train,:,m] = rs_misa_train[m].detach().numpy()
                    res_rs_misa_valid[ind_valid,:,m] = rs_misa_valid[m].detach().numpy()
                    res_rs_misa_test[ind_test,:,m] = rs_misa_test[m].detach().numpy()
        
        if e % epoch_interval == 0:
            for m in range(n_modality):
                res_recovered_source[e].append({'train': res_rs_misa_train[:,:,m], 'valid': res_rs_misa_valid[:,:,m], 'test': res_rs_misa_test[:,:,m]})
            
            if experiment == 'sim':
                for m in range(n_modality):
                    res_mcc[e].append({'train': mean_corr_coef_per_segment(res_rs_misa_train[:,:,m], s_train[:,:,m], y_train), \
                                    'valid': mean_corr_coef_per_segment(res_rs_misa_valid[:,:,m], s_valid[:,:,m], y_valid), \
                                    'test': mean_corr_coef_per_segment(res_rs_misa_test[:,:,m], s_test[:,:,m], y_test)})
                    if m == n_modality - 1:
                        metric_train = MMSE(res_rs_misa_train, s_train, y_train)
                        metric_valid = MMSE(res_rs_misa_valid, s_valid, y_valid)
                        metric_test = MMSE(res_rs_misa_test, s_test, y_test)
                        res_metric[e] = {'train': metric_train, 'valid': metric_valid, 'test': metric_test}
                
                current_mcc_valid = np.mean([res_mcc[e][m]['valid'][0][0] for m in range(n_modality)])
                if e == 0:
                    best_mcc_valid = current_mcc_valid
                if current_mcc_valid >= best_mcc_valid:
                    ckpt_file_best = ckpt_file.replace('.pt', '_best.pt')
                    torch.save(res_misa_dict, ckpt_file_best)
                    loss_valid = np.mean(loss['valid'][e,:]) # average across segments
                    print(f"Epoch: {e}; Validation loss: {loss_valid}; Validation MCC: {current_mcc_valid}; Saved checkpoint to: {ckpt_file_best}")
                    best_mcc_valid = current_mcc_valid
                    best_epoch = e

    # prepare output
    Results = {
        'loss': loss,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'mcc': res_mcc,
        'metric': res_metric,
        'best_epoch': best_epoch
    }

    return Results


def run_icebeem(args, config, method="icebeem"):
    # wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    cuda = config.cuda
    device = config.device
    dataset = config.dataset
    n_layer = args.n_layer if args.n_layer else config.n_layer
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = args.n_obs_per_seg if args.n_obs_per_seg else config.n_obs_per_seg
    
    lr_flow = config.icebeem.lr_flow
    lr_ebm = config.icebeem.lr_ebm
    n_layer_flow = config.icebeem.n_layer_flow
    ebm_hidden_size = config.icebeem.ebm_hidden_size
    
    epoch_interval = 10
    best_epoch = 0
    loss = {'train':np.zeros((n_epoch, n_modality)), 'valid':np.zeros((n_epoch, n_modality)), 'test':np.zeros((n_epoch, n_modality))}
    res_mcc = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_metric = {e: {} for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    res_recovered_source = {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1)]}
    best_epoch = np.zeros(n_modality)

    if experiment == "sim":
        # generate synthetic data
        x, y, s = generate_synthetic_data(latent_dim, n_segment, n_obs_per_seg*3, n_layer, seed=data_seed,
            simulationMethod=dataset, one_hot_labels=True, varyMean=False)
        
        x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n_obs_per_seg)

        res_ground_truth_source = {'train': s_train, 'valid': s_valid, 'test': s_test}

    elif experiment == "img":
        data = sio.loadmat(data_path)
        
        x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test  = split_img_data(data)

        res_ground_truth_source = {}
    
    n_layer_ebm = n_layer + 1
    print(f'Running {method} experiment with L={n_layer}; n_obs_per_seg={n_obs_per_seg}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; ayerebm={n_layer_ebm}; layerflow={n_layer_flow}; lrebm={lr_ebm}; lrflow={lr_flow}')

    # intiate one model for each modality
    for m in range(n_modality):
        ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{n_layer}_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}_epoch{n_epoch}_modality{m+1}_layerebm{n_layer_ebm}_layerflow{n_layer_flow}_lrebm{lr_ebm}_lrflow{lr_flow}.pt')
        
        # initialize iVAE weights with multi-view ICA initial sources from PCA and perm ICA
        fname = os.path.join(args.run, f"res_mvica_source{latent_dim}_obs{n_obs_per_seg}_seg{n_segment}_seed{seed}.p")
        if os.path.exists(fname):
            print(f'Loading multi-view ICA weight {fname}')
            res = pickle.load(open(fname, 'rb'))
            s_init_mvica = res['initial_recovered_source_per_modality'][m]['train']
        else:
            print(f'File {fname} not found. Initialize iVAE using random weights.')

        if experiment == "sim":
            res_dict = ICEBEEM_wrapper_(X=x_train[:,:,m].astype(np.float32), Xv=x_valid[:,:,m].astype(np.float32), Xt=x_test[:,:,m].astype(np.float32),
                                        Y=y_train.astype(np.float32), Yv=y_valid.astype(np.float32), Yt=y_test.astype(np.float32),
                                        S=s_train[:,:,m].astype(np.float32), Sv=s_valid[:,:,m].astype(np.float32), St=s_test[:,:,m].astype(np.float32), 
                                        Sinit=s_init_mvica.astype(np.float32),
                                        ebm_hidden_size=ebm_hidden_size, n_layer_ebm=n_layer_ebm, 
                                        n_layer_flow=n_layer_flow, lr_flow=lr_flow, lr_ebm=lr_ebm, 
                                        seed=seed, ckpt_file=ckpt_file, n_epoch=n_epoch)
        elif experiment == "img":
            res_dict = ICEBEEM_wrapper_(X=x_train[:,:,m].astype(np.float32), Xv=x_valid[:,:,m].astype(np.float32), Xt=x_test[:,:,m].astype(np.float32), 
                                        Y=y_train.astype(np.float32), Yv=y_valid.astype(np.float32), Yt=y_test.astype(np.float32), 
                                        Sinit=s_init_mvica.astype(np.float32),
                                        ebm_hidden_size=ebm_hidden_size, n_layer_ebm=n_layer_ebm, 
                                        n_layer_flow=n_layer_flow, lr_flow=lr_flow, lr_ebm=lr_ebm, 
                                        seed=seed, ckpt_file=ckpt_file, n_epoch=n_epoch)
        
        best_epoch[m] = res_dict['best_epoch']

        # store results
        for e in range(n_epoch):
            loss['train'][e,m] = res_dict['loss']['train'][e]
            loss['valid'][e,m] = res_dict['loss']['valid'][e]
            loss['test'][e,m] = res_dict['loss']['test'][e]
            if e % epoch_interval == 0:
                res_recovered_source[e].append({'train': res_dict['res_recovered_source'][e]['train'], 'valid': res_dict['res_recovered_source'][e]['valid'], 'test': res_dict['res_recovered_source'][e]['test']})
                if experiment == 'sim':
                    res_mcc[e].append({'train': res_dict['res_mcc'][e]['train'], 'valid': res_dict['res_mcc'][e]['valid'], 'test': res_dict['res_mcc'][e]['test']})
                    if m == n_modality - 1:
                        metric_train = MMSE(np.dstack([r['train'] for r in res_recovered_source[e]]), s_train, y_train)
                        metric_valid = MMSE(np.dstack([r['valid'] for r in res_recovered_source[e]]), s_valid, y_valid)
                        metric_test = MMSE(np.dstack([r['test'] for r in res_recovered_source[e]]), s_test, y_test)
                        res_metric[e] = {'train': metric_train, 'valid': metric_valid, 'test': metric_test}
    
    # prepare output
    Results = {
        'loss': loss,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'mcc': res_mcc,
        'metric': res_metric,
        'best_epoch': best_epoch
    }

    return Results


def run_mvica(args, config, method="mvica"):
    # wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    dataset = config.dataset
    n_layer = args.n_layer if args.n_layer else config.n_layer
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = args.n_obs_per_seg if args.n_obs_per_seg else config.n_obs_per_seg

    if experiment == "sim":
        # generate synthetic data
        x, y, s = generate_synthetic_data(latent_dim, n_segment, n_obs_per_seg*3, n_layer, seed=data_seed,
            simulationMethod=dataset, one_hot_labels=True, varyMean=False)
        
        x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n_obs_per_seg)
        
        res_ground_truth_source = {'train': s_train, 'valid': s_valid, 'test': s_test}

    elif experiment == "img":
        data = sio.loadmat(data_path)
        
        x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test  = split_img_data(data)

        res_ground_truth_source = {}
    
    print(f'Running {method} experiment with L={n_layer}; n_obs_per_seg={n_obs_per_seg}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}')
    
    key_list = ['train', 'valid', 'test']
    x_dict = {'train': x_train, 'valid': x_valid, 'test': x_test}
    x_rs_dict = {}
    s_rec_dict = {}
    res_wp = {}
    res_w0p = {}
    res_recovered_source = []
    res_initial_recovered_source_per_modality = []
    res_recovered_source_per_modality = []
    res_mcc = []
    res_metric = {}
    
    init = np.stack([np.eye(latent_dim) for _ in range(n_modality)]) # identity initialization
    init += np.random.uniform(-0.1,0.1,(n_modality, latent_dim, latent_dim)) # random initialization

    for i, key in enumerate(key_list):
        x_set = x_dict[key]
        n_group = x_set.shape[2]
        n_feature = x_set.shape[1]
        n_sample = x_set.shape[0]
        x_rs = np.zeros((n_group, n_feature, n_sample))

        for i in range(n_group):
            for j in range(n_feature):
                for k in range(n_sample):
                    x_rs[i, j, k] = x_set[k, j, i]
        x_rs_dict[key] = x_rs

        p, w0, w, s_rec = multiviewica(x_rs, n_components=latent_dim, tol=1e-4, max_iter=10000, init=init, random_state=seed)
        s_rec_dict[key] = s_rec.T
        res_w0p[key] = np.stack([w0[m,:,:] @ p[m,:,:] for m in range(n_modality)])
        res_wp[key] = np.stack([w[m,:,:] @ p[m,:,:] for m in range(n_modality)])
        
    # store results
    for m in range(n_modality):
        res_recovered_source.append(s_rec_dict)
        res_initial_recovered_source_per_modality.append({'train':(res_w0p['train'][m,:,:]@x_rs_dict['train'][m,:,:]).T,\
                                                          'valid':(res_w0p['valid'][m,:,:]@x_rs_dict['valid'][m,:,:]).T,\
                                                          'test':(res_w0p['test'][m,:,:]@x_rs_dict['test'][m,:,:]).T})
        res_recovered_source_per_modality.append({'train':(res_wp['train'][m,:,:]@x_rs_dict['train'][m,:,:]).T,\
                                                  'valid':(res_wp['valid'][m,:,:]@x_rs_dict['valid'][m,:,:]).T,\
                                                  'test':(res_wp['test'][m,:,:]@x_rs_dict['test'][m,:,:]).T})
    
    if experiment == "sim":
        for m in range(n_modality):
            res_mcc.append({'train': mean_corr_coef_per_segment(s_rec_dict['train'], s_train[:,:,m], y_train), \
                            'valid': mean_corr_coef_per_segment(s_rec_dict['valid'], s_valid[:,:,m], y_valid), \
                            'test': mean_corr_coef_per_segment(s_rec_dict['test'], s_test[:,:,m], y_test)})
        metric_train = MMSE(np.dstack([r['train'] for r in res_recovered_source]), s_train, y_train)
        metric_valid = MMSE(np.dstack([r['valid'] for r in res_recovered_source]), s_valid, y_valid)
        metric_test = MMSE(np.dstack([r['test'] for r in res_recovered_source]), s_test, y_test)
        res_metric = {'train': metric_train, 'valid': metric_valid, 'test': metric_test}

    # prepare output
    Results = {
        'recovered_source': res_recovered_source,
        'recovered_source_per_modality': res_recovered_source_per_modality,
        'initial_recovered_source_per_modality': res_initial_recovered_source_per_modality,
        'ground_truth_source': res_ground_truth_source,
        'weight': res_wp,
        'weight_init': res_w0p,
        'x': x_rs_dict,
        'mcc': res_mcc,
        'metric': res_metric,
    }

    return Results