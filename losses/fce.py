### Pytorch implementation of training EBMs via FCE
#
#
import contextlib
import numpy as np
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.imca import ContrastiveConditionalDataset, SimpleDataset, ConditionalDataset
from data.utils import to_one_hot
from sklearn.decomposition import FastICA
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
from metrics.mcc import mean_corr_coef_per_segment
from model.utils import EarlyStopper

class ConditionalFCE(object):
    """
    train an energy based model using noise contrastive estimation
    where we assume we observe data from multiple segments/classes
    this is useful for nonlinear ICA and semi supervised learning !
    """

    def __init__(self, data, segments, energy_MLP, flow_model, verbose=False, data_valid=None, segments_valid=None, data_test=None, segments_test=None, source=None, source_valid=None, source_test=None):
        self.data = data
        self.segments = segments
        self.source = source
        self.data_valid = data_valid
        self.segments_valid = segments_valid
        self.source_valid = source_valid
        self.data_test = data_test
        self.segments_test = segments_test
        self.source_test = source_test
        self.contrast_segments = (np.ones(self.segments.shape) / self.segments.shape[1]).astype(np.float32)
        self.contrast_segments_valid = (np.ones(self.segments_valid.shape) / self.segments_valid.shape[1]).astype(np.float32)
        self.contrast_segments_test = (np.ones(self.segments_test.shape) / self.segments_test.shape[1]).astype(np.float32)
        self.energy_MLP = energy_MLP
        self.ebm_norm = -5.
        self.hidden_dim = self.energy_MLP.linearLast.weight.shape[0]
        self.n_segments = self.segments.shape[1]
        self.ebm_finalLayer = torch.tensor(np.ones((self.hidden_dim, self.n_segments)).astype(np.float32))
        # self.ebm_finalLayer = torch.tensor( np.random.random(( self.hidden_dim, self.n_segments )).astype(np.float32) )
        self.flow_model = flow_model  # flow model, must have sample and log density capabilities
        self.noise_samples = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose

    def sample_noise(self, n):
        if self.device == 'cuda':
            return self.flow_model.module.sample(n)[-1].detach().cpu().numpy()
        else:
            return self.flow_model.sample(n)[-1].detach().numpy()

    def noise_logpdf(self, dat):
        """
        compute log density under flow model
        """
        zs, prior_logprob, log_det = self.flow_model(dat)
        flow_logdensity = (prior_logprob + log_det)
        return flow_logdensity

    def compute_ebm_logpdf(self, dat, seg, logNorm, augment=False):
        act_allLayer = torch.mm(self.energy_MLP(dat), self.ebm_finalLayer)

        if augment:
            # we augment the feature extractor
            act_allLayer += torch.mm(self.energy_MLP(dat) * self.energy_MLP(dat),
                                     self.ebm_finalLayer * self.ebm_finalLayer)

        # now select relevant layers by multiplying by mask matrix and reducing (and adding log norm)
        act_segment = (act_allLayer * seg).sum(1) + logNorm

        return act_segment

    def pretrain_ebm_model(self, source_init, batch_size=256, max_iter=1000, cuda=True):
        
        # load data
        dset = ConditionalDataset(X=self.data, Y=self.segments, S=source_init, device=self.device)
        loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        data_loader = DataLoader(dset, shuffle=False, batch_size=batch_size, **loader_params)
        
        loss = torch.nn.MSELoss()
        
        # train encoder to approximate initial sources
        norm_const_s = np.max(np.abs(source_init))
        optimizer_s = optim.Adam(self.energy_MLP.parameters(), lr=1e-3)
        scheduler_s = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s, factor=0.1, patience=20, verbose=True)
        early_stopper = EarlyStopper()

        self.energy_MLP.to(self.device)
        self.energy_MLP.train()
        
        for it in range(max_iter):
            loss_total = 0
            for _, (x, u, s) in enumerate(data_loader):
                optimizer_s.zero_grad()
                rs = self.energy_MLP.forward(x.to(self.device))
                loss_batch = loss(rs, s) / norm_const_s
                loss_batch.backward()
                optimizer_s.step()
                loss_total += loss_batch.item()
            loss_total /= len(data_loader)
            scheduler_s.step(loss_total)
            print(f'EBM initialization - source reconstruction - iteration {it}; training loss: {loss_total:.3f}')
            if early_stopper.early_stop(loss_total):
                print(f'Early stopping triggered!')
                break
    
    def train_ebm_fce(self, epochs=500, epoch_interval=10, lr=.0001, ckpt_file='ebm_fce.pt', cutoff=None, augment=False, finalLayerOnly=False, useVAT=False):
        """
        FCE training of EBM model
        """
        if self.verbose:
            print('Training energy based model using FCE' + useVAT * ' with VAT penalty')

        if cutoff is None:
            cutoff = 1.00  # will basically only stop with perfect classification
        
        # sample noise data
        n = self.data.shape[0]
        self.noise_samples = self.sample_noise(n)  # self.noise_dist.sample( n )
        # define classification labels
        y = np.array([0] * n + [1] * n)
        # define
        dat_fce = ContrastiveConditionalDataset(np.vstack((self.data, self.noise_samples)).astype(np.float32),
                                                to_one_hot(y)[0].astype(np.float32),
                                                np.vstack((self.segments, self.contrast_segments)), device=self.device)
        fce_loader = DataLoader(dat_fce, shuffle=True, batch_size=128)

        # validation
        n_valid = self.data_valid.shape[0]
        noise_valid = self.sample_noise(n_valid)
        y_valid = np.array([0] * n_valid + [1] * n_valid)
        dat_fce_valid = ContrastiveConditionalDataset(np.vstack((self.data_valid, noise_valid)).astype(np.float32),
                                                    to_one_hot(y_valid)[0].astype(np.float32),
                                                    np.vstack((self.segments_valid, self.contrast_segments_valid)), device=self.device)
        fce_loader_valid = DataLoader(dat_fce_valid, shuffle=False, batch_size=128)

        # test
        n_test = self.data_test.shape[0]
        noise_test = self.sample_noise(n_test)
        y_test = np.array([0] * n_test + [1] * n_test)
        dat_fce_test = ContrastiveConditionalDataset(np.vstack((self.data_test, noise_test)).astype(np.float32),
                                                    to_one_hot(y_test)[0].astype(np.float32),
                                                    np.vstack((self.segments_test, self.contrast_segments_test)), device=self.device)
        fce_loader_test = DataLoader(dat_fce_test, shuffle=False, batch_size=128)

        # define log normalization constant
        ebm_norm = self.ebm_norm  # -5.
        logNorm = torch.from_numpy(np.array(ebm_norm).astype(np.float32)).float().to(self.device)
        logNorm.requires_grad_()

        self.ebm_finalLayer.requires_grad_()

        # define optimizer
        if finalLayerOnly:
            # only train the final layer, this is the equivalent of g(y) in IMCA manuscript.
            optimizer = optim.Adam([self.ebm_finalLayer] + [logNorm], lr=lr)
        else:
            optimizer = optim.Adam(list(self.energy_MLP.parameters()) + [self.ebm_finalLayer] + [logNorm], lr=lr)

        self.energy_MLP.to(self.device)
        self.energy_MLP.train()

        # begin optimization
        loss_criterion = nn.BCELoss()

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.energy_MLP.cuda()
            self.energy_MLP = torch.nn.DataParallel(self.energy_MLP, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        num_correct = {'train':np.zeros(epochs), 'valid':np.zeros(epochs), 'test':np.zeros(epochs)}
        loss_total = {'train':np.zeros(epochs), 'valid':np.zeros(epochs), 'test':np.zeros(epochs)}
        res_recovered_source = {e: [] for e in [n*epoch_interval for n in range(epochs//epoch_interval+1)]}
        res_mcc = {e: [] for e in [n*epoch_interval for n in range(epochs//epoch_interval+1)]}
        best_epoch = 0
        
        for e in range(epochs):
            # training
            for _, (dat, label, seg) in enumerate(fce_loader):
                # consider adding VAT loss
                if useVAT:
                    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                    lds = vat_loss(self.energy_MLP, dat)

                # noise model probs:
                noise_logpdf = self.noise_logpdf(dat).view(-1,1) # torch.tensor( self.noise_dist.logpdf( dat ).astype(np.float32) ).view(-1,1)

                # pass to correct device:
                if use_cuda:
                    dat = dat.to(self.device)
                    seg = seg.to(self.device)
                    label = label.to(self.device)
                # dat, seg = dat.cuda(), seg.cuda()

                # get ebm log pdf
                ebm_logpdf = self.compute_ebm_logpdf(dat, seg, logNorm, augment=augment).view(-1, 1)

                # define logits
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
                logits.to(self.device)

                # compute accuracy:
                num_correct['train'][e] += (logits.argmax(1) == label.argmax(1)).sum().item()

                # define loss
                loss = loss_criterion(torch.sigmoid(logits), label)
                if useVAT:
                    loss += 1 * lds

                loss_total['train'][e] += loss.item()

                # take gradient step
                self.energy_MLP.zero_grad()

                optimizer.zero_grad()

                # compute gradients
                loss.backward()

                # update parameters
                optimizer.step()

            # validation
            for _, (dat, label, seg) in enumerate(fce_loader_valid):
                # consider adding VAT loss
                if useVAT:
                    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                    lds = vat_loss(self.energy_MLP, dat)
                noise_logpdf = self.noise_logpdf(dat).view(-1,1)
                if use_cuda:
                    dat = dat.to(self.device)
                    seg = seg.to(self.device)
                    label = label.to(self.device)
                ebm_logpdf = self.compute_ebm_logpdf(dat, seg, logNorm, augment=augment).view(-1, 1)
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
                logits.to(self.device)
                num_correct['valid'][e] += (logits.argmax(1) == label.argmax(1)).sum().item()
                loss = loss_criterion(torch.sigmoid(logits), label)
                if useVAT:
                    loss += 1 * lds
                loss_total['valid'][e] += loss.item()
            
            # testing
            for _, (dat, label, seg) in enumerate(fce_loader_test):
                # consider adding VAT loss
                if useVAT:
                    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
                    lds = vat_loss(self.energy_MLP, dat)
                noise_logpdf = self.noise_logpdf(dat).view(-1,1)
                if use_cuda:
                    dat = dat.to(self.device)
                    seg = seg.to(self.device)
                    label = label.to(self.device)
                ebm_logpdf = self.compute_ebm_logpdf(dat, seg, logNorm, augment=augment).view(-1, 1)
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
                logits.to(self.device)
                num_correct['test'][e] += (logits.argmax(1) == label.argmax(1)).sum().item()
                loss = loss_criterion(torch.sigmoid(logits), label)
                if useVAT:
                    loss += 1 * lds
                loss_total['test'][e] += loss.item()

            torch.save({'ebm_mlp': self.energy_MLP.state_dict(),
                        'ebm_finalLayer': self.ebm_finalLayer,
                        'flow': self.flow_model.state_dict()}, ckpt_file)
            
            if e % epoch_interval == 0:
                rs_train = self.unmixSamples(self.data, modelChoice='ebm')
                rs_ica_train = FastICA().fit_transform((rs_train))
                rs_valid = self.unmixSamples(self.data_valid, modelChoice='ebm')
                rs_ica_valid = FastICA().fit_transform((rs_valid))
                rs_test = self.unmixSamples(self.data_test, modelChoice='ebm')
                rs_ica_test = FastICA().fit_transform((rs_test))
                res_recovered_source[e] = {'train': rs_ica_train, 'valid': rs_ica_valid, 'test': rs_ica_test}
                if self.source is not None:
                    res_mcc[e] = {'train': mean_corr_coef_per_segment(rs_ica_train, self.source, self.segments), \
                                  'valid': mean_corr_coef_per_segment(rs_ica_valid, self.source_valid, self.segments_valid), \
                                  'test': mean_corr_coef_per_segment(rs_ica_test, self.source_test, self.segments_test)}
                    
                    current_mcc_valid = res_mcc[e]['valid'][0][0]
                    if e == 0:
                        best_mcc_valid = current_mcc_valid
                    if current_mcc_valid >= best_mcc_valid:
                        ckpt_file_best = ckpt_file.replace('.pt', '_best.pt')
                        shutil.copy2(ckpt_file, ckpt_file_best)
                        loss_valid = loss_total['valid'][e]
                        print(f"Epoch: {e}; Validation loss: {loss_valid}; Validation MCC: {current_mcc_valid}; Saved checkpoint to: {ckpt_file_best}")
                        best_mcc_valid = current_mcc_valid
                        best_epoch = e
            
            # print some statistics
            if self.verbose:
                print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_total[e]['train'], 4),
                                                                 np.round(num_correct[e]['train'] / (2 * n), 3)))
            # if num_correct[e]['train'] / (2 * n) > cutoff:
            #     # stop training
            #     if self.verbose:
            #         print('epoch {}\taccuracy: {}'.format(e, np.round(num_correct[e]['train'] / (2 * n), 3)))
            #         print('cutoff value satisfied .. stopping training\n----------\n')
            #     break

        self.ebm_norm = logNorm.item()
        
        res_dict = {'loss': loss_total,
                    'num_correct': num_correct,
                    'res_recovered_source': res_recovered_source,
                    'res_mcc': res_mcc,
                    'best_epoch': best_epoch
                    }
    
        return res_dict
    
    def reset_noise(self):
        self.noise_samples = self.sample_noise(self.noise_samples.shape[0])

    def pretrain_flow_model(self, epochs=50, lr=1e-4):
        """
        pertraining of flow model using MLE
        """
        optimizer = optim.Adam(self.flow_model.parameters(), lr=1e-4, weight_decay=1e-5)  # todo tune WD
        # print("number of params: ", sum(p.numel() for p in model_flow.parameters()))

        dset = SimpleDataset(self.data.astype(np.float32), device=self.device)
        train_loader = DataLoader(dset, shuffle=True, batch_size=128)

        # run optimization
        loss_vals = []

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.flow_model.to(self.device)
            self.flow_model = torch.nn.DataParallel(self.flow_model, device_ids=range(torch.cuda.device_count()))
            # self.flow_model.to( self.device )
            cudnn.benchmark = True
            print("using gpus! " + str(self.device))

        self.flow_model.train()
        for e in range(epochs):
            loss_val = 0
            for _, dat in enumerate(train_loader):
                if use_cuda:
                    dat = dat.cuda()
                    dat = Variable(dat)
                zs, prior_logprob, log_det = self.flow_model(dat)
                logprob = prior_logprob + log_det
                loss = - torch.sum(logprob)  # NLL

                # print(loss.item())
                loss_val += loss.item()

                #
                self.flow_model.zero_grad()
                optimizer.zero_grad()

                # compute gradients
                loss.backward()

                # update parameters
                optimizer.step()

            if self.verbose:
                print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
            loss_vals.append(loss_val)

    def train_flow_fce(self, epochs=50, lr=1e-4, objConstant=-1.0, cutoff=None):
        """
        FCE training of EBM model
        """
        if self.verbose:
            print('Training flow contrastive noise for FCE')

        if cutoff is None:
            cutoff = 0.  # basically only stop for perfect misclassification

        # noise data already sampled during EBM training
        n = self.data.shape[0]
        self.reset_noise()
        # define classification labels
        y = np.array([0] * n + [1] * n)

        # define
        dat_fce = ContrastiveConditionalDataset(np.vstack((self.data, self.noise_samples)).astype(np.float32),
                                                to_one_hot(y)[0].astype(np.float32),
                                                np.vstack((self.segments, self.segments)), device=self.device)
        fce_loader = DataLoader(dat_fce, shuffle=True, batch_size=128)

        # define optimizer
        optimizer = optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=1e-5)  # todo tune WD

        use_cuda = torch.cuda.is_available()
        self.flow_model.to(self.device)
        self.flow_model.train()

        # begin optimization
        loss_criterion = nn.BCELoss()

        for e in range(epochs):
            num_correct = 0
            loss_val = 0
            for _, (dat, label, seg) in enumerate(fce_loader):
                # pass to correct device:
                if use_cuda:
                    dat = dat.to(self.device)
                    seg = seg.to(self.device)
                    label = label.to(self.device)

                # noise model probs:
                noise_logpdf = self.noise_logpdf(dat).view(-1,
                                                           1)  # torch.tensor( self.noise_dist.logpdf( dat ).astype(np.float32) ).view(-1,1)

                # get ebm model probs:
                ebm_logpdf = self.compute_ebm_logpdf(dat, seg, self.ebm_norm).view(-1, 1)

                # define logits
                logits = torch.cat((ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
                logits *= objConstant

                # compute accuracy:
                num_correct += (logits.argmax(1) == label.argmax(1)).sum().item()

                # define loss
                loss = loss_criterion(torch.sigmoid(logits), label)
                loss_mle = - torch.mean(noise_logpdf)  # mle objective for training data
                loss_val += (loss.item() + loss_mle.item())  # this is the jensen shannon

                # take gradient step
                self.flow_model.zero_grad()

                optimizer.zero_grad()

                # compute gradients
                loss.backward()

                # update parameters
                optimizer.step()

            # print some statistics
            if self.verbose:
                print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_val, 4),
                                                                 np.round(1 - num_correct / (2 * n), 3)))
            if 1 - num_correct / (2 * n) < cutoff:
                if self.verbose:
                    print('epoch {}\taccuracy: {}'.format(e, np.round(1 - num_correct / (2 * n), 3)))
                    print('cutoff value satisfied .. stopping training\n----------\n')
                break

    def unmixSamples(self, data, modelChoice):
        """
        perform unmixing of samples
        """
        if modelChoice == 'EBM':
            # unmix using EBM:
            if self.device == 'gpu':
                recov = self.energy_MLP(torch.tensor(data.astype(np.float32))).detach().numpy()
            else:
                recov = self.energy_MLP(torch.tensor(data.astype(np.float32))).detach().cpu().numpy()
        else:
            # unmix using flow model
            if self.device == 'cpu':
                recov = self.flow_model(torch.tensor(data.astype(np.float32)))[0][-1].detach().numpy()
            else:
                recov = self.flow_model(torch.tensor(data.astype(np.float32)))[0][-1].detach().cpu().numpy()

        return recov


### Virtual adversarial regularization loss
#
#
# this code has been shamelessly taken from:
#  https://raw.githubusercontent.com/lyakaap/VAT-pytorch/master/vat.py
#
###


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
