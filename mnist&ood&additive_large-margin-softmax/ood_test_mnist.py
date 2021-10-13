
import os
import torch
import torchvision
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from models import MNISTNet
import numpy as np
import argparse
from tqdm import tqdm
from loguru import logger

from data_loader import get_clean_data_loader
from exp import get_test_transform, get_std, get_num_labels
from metrics import get_metrics
import matplotlib.pyplot as plt
import seaborn as sns
# NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
#       see, https://github.com/pytorch/vision/issues/3497 for more information
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


# ---- 这个用于mnist的ood
def ece_score(py, y_test, dir_name,n_bins=10,):
    # py = np.array(py)
    # y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    # names = ['0-0.1', '0.1-0.2', '0.2-0.3',
    #          '0.3-0.4', '0.1-0.2', '0.1-0.2',
    #          '0.6-0.7', '0.7-0.8', '0.1-0.2', '0.1-0.2']
    # plt.figure(figsize=(9, 3))
    ece=ece / sum(Bm)
    plt.figure()
    x = np.arange(0.05,1.05,0.1)
    plt.plot(x,acc,'r',label='Accuracy')
    plt.plot(x, conf, 'b', label='Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy vs Confidence')
    plt.legend()
    plt.title("ECE: %.3f" % ece)
    plt.savefig(dir_name+ '_ece.png')
    plt.close()
        # print()
    return ece



def get_energy_score(model, data_loader,device):
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            _scores, _ = torch.max(logits, dim=1)
            scores.append(_scores.cpu().numpy())
    scores = np.concatenate(scores)
    return scores

def get_free_energy_score(model, data_loader,device):
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            _scores= torch.logsumexp(logits, dim=1)
            scores.append(_scores.cpu().numpy())
    scores = np.concatenate(scores)
    return scores

def get_base_score(model, data_loader,device):
    '''
    Baseline, calculating Maxium Softmax Probability as prediction confidence
    Reference: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks 
    Link: https://arxiv.org/abs/1610.02136
    '''
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores

def plot_likelihood(org_probs,targets_hot,dir_name):
    like_mat = org_probs * targets_hot.float()  # num_samples, num_classes
    kwargs = dict(alpha=0.5, bins=20)
    # from matplotlib.backends.backend_pdf import PdfPages
    # pdf = PdfPages('iNaturalist2018_90' + right_names[j] + '_test_likelihood.pdf')
    fig, ax = plt.subplots(1, 1,)
    ax.set_yticks([0, 10000 * 0.10, 10000 * 1])
    ax.set_xlabel('likelihood')
    ax.hist(torch.sum(like_mat, dim=1).cpu().numpy(),**kwargs, label='likelihood distribution',)  # num_samples
    ax.legend(loc='upper center')
    fig.savefig(dir_name + '_' + 'likelihood.png')
    # f = open(dir_name + '/'+'dis', 'wb')
    # import pickle
    # pickle.dump(like_mat, f)
    # pdf.savefig()
    plt.close()
    # pdf.close()

def plot_margin(total_logits,targets_hot,dir_name):
    logits_mat = total_logits * targets_hot.float()  # num_samples, num_classes
    max_negative_logits, _=torch.max(total_logits * (torch.ones_like(targets_hot)-targets_hot).float(),dim=1) # num_samples
    target_logits=torch.sum(logits_mat, dim=1)
    margin_mat = (target_logits-max_negative_logits).cpu().numpy()  # num_samples each with a margin
    kwargs = dict(alpha=0.5, bins=20)
    fig, ax = plt.subplots(1, 1,)
    ax.set_xlabel('margin')
    ax.hist(margin_mat,**kwargs, label='margin distribution',)  # num_samples
    ax.legend(loc='upper left',) # fontdict={'fontsize': 25}
    fig.savefig(dir_name + '_' + 'margin.png')
    plt.close()

def stat_energy(total_logits,targets_hot,dir_name):
    # num_classes=total_logits.size()[-1]
    logits_mat = total_logits * targets_hot.float()  # num_samples, num_classes
    max_negative_logits,_=torch.max(total_logits * (torch.ones_like(targets_hot)-targets_hot).float(),dim=1) # num_samples
    target_logits=torch.sum(logits_mat, dim=1)
    print('energy of target: ',torch.mean(-target_logits),'| energy of the best other: ',torch.mean(-max_negative_logits))



def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--gpu', default=0,
                        type=int)
    parser.add_argument('--eval_type', default='ood',
                    type=str, choices=['acc', 'ood', 'calibration'])
    parser.add_argument('--batch_size', default=100, type=int, help='training batch size')
    parser.add_argument('--model_path', type=str, default='../encourage_models_zhao/cifar10_resnet50_baseline_ls.pth', help='trained model path')
    parser.add_argument('--log_file', type=str, default='./log/test_default.log')
    parser.add_argument('--model', default='lenet',type=str, required=False, help='Model Type')
    parser.add_argument('--ind', default='cifar10', type=str, choices=['mnist','fmnist','cifar10', 'cifar100','kmnist'])
    parser.add_argument('--ood', default='svhn', type=str, help='ood dataset')
    parser.add_argument('--output_name', type=str, default=None)
    parser.add_argument('--ood_measure', type=str, default='cond_energy')
    args = parser.parse_args()
    args.dataset = args.ind
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    batch_size = args.batch_size

    device = 'cuda:%d'%args.gpu
    # Data
    logger.info('==> Preparing data..')
    # transform_test = get_test_transform(args.ind)
    # std = get_std(args.ind)
    if args.ind =='mnist':
        kwargs = {'num_workers': 4, 'pin_memory': True}
        transform_test=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ind_test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
    else:
        transform_test = transforms.Compose([
                            transforms.Resize(int(32 * 256 / 224)),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
                         ])

        ind_test_loader = get_clean_data_loader(args.ind, batch_size, TF=transform_test, train=False)
    ood_test_loader = get_clean_data_loader(args.ood, batch_size, TF=transform_test, train=False)

    # Models
    logger.info('==> Building model..')
    #model = get_model(args)
    # model = torchvision.models.__dict__['resnet50'](num_classes=get_num_labels(args.ind))
    model = MNISTNet(margin=1, device=device).to(device)
    weight = torch.load(args.model_path)
    if type(weight) == dict and "model" in weight.keys():
        weight = weight['model']
    
    model.load_state_dict(weight)
    logger.info("model loaded from {}".format(args.model_path))
    model = model.to(device)
    # 1. ece
    model.eval()
    total_logits = torch.empty((0, 10)).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    with torch.no_grad():
        # for i, (images, target) in enumerate(val_loader):
        for imgs, labels in tqdm(ind_test_loader):
            imgs = imgs.to(device)
            labels=labels.to(device)
            logits = model(imgs)
            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, labels))
    org_probs = F.softmax(total_logits, dim=1)
    targets_hot = F.one_hot(total_labels.view(-1), num_classes=10)  # bsz, num_classes
    dir_name=args.output_name
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # 1 likelihood
    plot_likelihood(org_probs, targets_hot, dir_name)
    # 2 ece
    ece = ece_score(org_probs.cpu().numpy(), total_labels.cpu().numpy(), dir_name=args.output_name,)
    print('uncalibrated_score:',ece)
    # 3 margin distribution
    plot_margin(total_logits, targets_hot, dir_name)
    # 4 stat energy
    stat_energy(total_logits, targets_hot, dir_name)
    # 5. ood
    if args.ood !='none':
        # if args.ood_measure == 'cond_energy':
        #     ind_scores = get_energy_score(model, ind_test_loader,device)
        #     ood_scores = get_energy_score(model, ood_test_loader,device)
        # elif args.ood_measure=='free_energy':
        #     ind_scores = get_free_energy_score(model, ind_test_loader,device)
        #     ood_scores = get_free_energy_score(model, ood_test_loader,device)
        # else:
        #     ind_scores = get_base_score(model, ind_test_loader,device)
        #     ood_scores = get_base_score(model, ood_test_loader,device)
        for args.ood_measure in ['cond_energy','free_energy','prob']:
            if args.ood_measure == 'cond_energy':
                ind_scores = get_energy_score(model, ind_test_loader,device)
                ood_scores = get_energy_score(model, ood_test_loader,device)
            elif args.ood_measure=='free_energy':
                ind_scores = get_free_energy_score(model, ind_test_loader,device)
                ood_scores = get_free_energy_score(model, ood_test_loader,device)
            else:
                ind_scores = get_base_score(model, ind_test_loader,device)
                ood_scores = get_base_score(model, ood_test_loader,device)
            logger.info("ind scores shape:{}".format(str(ind_scores.shape)))
            logger.info("ood scores shape:{}".format(str(ood_scores.shape)))
            metrics = get_metrics(ind_scores, ood_scores)
            logger.info(str(metrics))
            if args.output_name:

                sns.kdeplot(ind_scores,cut=0,shade=True, label=args.ind+'(ID)')
                sns.kdeplot(ood_scores,cut=0,shade=True, label=args.ood+'(OOD)')
                if args.ood_measure in ['energy', 'cond_energy']:
                    plt.xlabel('Negative Max Conditional Energy')
                    print('Negative Max Conditional Energy')
                elif args.ood_measure == 'free_energy':
                    plt.xlabel('Negative Free Energy')
                    print('Negative Free Energy')
                else:  # prob
                    plt.xlabel('Max Probability')
                    print('Max Probability')
                plt.legend(loc='upper right')
                for metric_i in metrics.keys():
                    print('{} vs {}, {} = {:.2%} '.format(args.ind, args.ood, metric_i, float(metrics[metric_i])))
                plt.title('{} vs {}, FPR95 = {:.2%} '.format(args.ind, args.ood, float(metrics['FPR@tpr=0.95'])))
                plt.savefig(args.output_name+'_'+args.ood_measure+'_ood.png')
                plt.close()

        

if __name__ == '__main__':
    main()


