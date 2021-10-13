
import os
import torch
import torchvision
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

import numpy as np
import argparse
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

# global device
# from .lib.utils.data_loader import get_clean_data_loader
# from .lib.utils.exp import get_test_transform, get_std, get_num_labels
# from .lib.utils.metrics import get_metrics
from metrics import get_metrics, _ECELoss
from load_iNaturalist18 import load_iNaturalist18_data


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

def get_ece(model, data_loader,device,dir_name,num_classes=1000):
    model.eval()
    total_logits = torch.empty((0, num_classes)).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    with torch.no_grad():
        # for i, (images, target) in enumerate(val_loader):
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            labels=labels.to(device)
            logits = model(imgs)
            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, labels))
    org_probs = F.softmax(total_logits, dim=1)
    ece = ece_score(org_probs.cpu().numpy(), total_labels.cpu().numpy(), dir_name=dir_name,)
    print('ECE',ece)
    return


def get_cond_energy_score(model, data_loader,device):
    model.eval()
    scores = []
    with torch.no_grad():
        # for i, (images, target) in enumerate(val_loader):
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            _scores, _ = torch.max(logits, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores

def get_cond_energy_score_inl(model, data_loader,device):
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, labels,_ in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            _scores, _ = torch.max(logits, dim=1)
            # _scores = torch.logsumexp(logits, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores

def get_free_energy_score(model, data_loader,device):
    model.eval()
    scores = []
    with torch.no_grad():
        # for i, (images, target) in enumerate(val_loader):
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            # _scores, _ = torch.max(logits, dim=1)
            _scores = torch.logsumexp(logits, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores

def get_free_energy_score_inl(model, data_loader,device):
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, labels,_ in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            _scores = torch.logsumexp(logits, dim=1)
            # _scores, _ = torch.max(logits, dim=1)
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
def get_base_score_inl(model, data_loader,device):
    '''
    Baseline, calculating Maxium Softmax Probability as prediction confidence
    Reference: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
    Link: https://arxiv.org/abs/1610.02136
    '''
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(data_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device', default='0,1,2,3',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--eval_type', default='ood',
                    type=str, choices=['acc', 'ood', 'calibration'])
    parser.add_argument('--batch_size', default=100, type=int, help='training batch size')
    # /home/xxx/ood_imagenet_inl/resnet50_baseline.pth.tar
    # /home/xxx/ood_imagenet_inl/resnet50_le075.pth.tar
    # /home/xxx/ood_imagenet_inl/resnet50_le1.pth.tar
    parser.add_argument('--model_path', type=str, default='../encourage_models/cifar10_resnet50_baseline_ls.pth', help='trained model path')
    parser.add_argument('--log_file', type=str, default='./log/test_default.log')
    parser.add_argument('--model', default='lenet',type=str, required=False, help='Model Type')
    parser.add_argument('--ind', default='ImageNet', type=str, choices=['mnist','fmnist','cifar10', 'cifar100','kmnist','ImageNet'])
    parser.add_argument('--ood', default='iNaturalist18', type=str, help='ood dataset')
    parser.add_argument('--output_image', type=str, default=None)
    parser.add_argument('--ood_measure', type=str, default='energy')
    args = parser.parse_args()
    args.dataset = args.ind
    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    batch_size = args.batch_size

    device ='cuda:%d'%args.gpu

    # Data
    logger.info('==> Preparing data..')
    # transform_test = get_test_transform(args.ind)
    # std = get_std(args.ind)
    
    # transform_test = transforms.Compose([
    #                     transforms.Resize(int(32 * 256 / 224)),
    #                     transforms.CenterCrop(32),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    #                  ])
    # imagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if args.ind=='ImageNet':
        ind_test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("/data/datasets/Img/val", transform_test),
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)
    else:
        ind_test_loader = get_clean_data_loader(args.ind, batch_size, TF=transform_test, train=False)
    if args.ood == 'iNaturalist18':
        ood_test_loader = load_iNaturalist18_data(batch_size, transform=transform_test,train=False,num_workers=16)
    else:
        ood_test_loader = get_clean_data_loader(args.ood, batch_size, TF=transform_test, train=False)

    # Models
    logger.info('==> Building model..')
    #model = get_model(args)
    model = torchvision.models.__dict__['resnet50'](num_classes=1000)
    weight = torch.load(args.model_path)
    # if type(weight) == dict and "model" in weight.keys():
    #     weight = weight['model']
    weight = weight['state_dict']
    # print('check keys',weight.keys())
    for k in list(weight.keys()):
        new_k = k.replace('module.','')
        weight[new_k] = weight[k]
        del weight[k]
    model.load_state_dict(weight)
    logger.info("model loaded from {}".format(args.model_path))
    model = model.to(device)
    get_ece(model,ind_test_loader,device,dir_name=args.output_image.replace('.png',''))

    if args.ood_measure == 'free_energy':
        ind_scores = get_free_energy_score(model, ind_test_loader,device)
        ood_scores = get_free_energy_score_inl(model, ood_test_loader,device)
    elif args.ood_measure == 'cond_energy':
        ind_scores = get_cond_energy_score(model, ind_test_loader,device)
        ood_scores = get_cond_energy_score_inl(model, ood_test_loader,device)
    else:
        ind_scores = get_base_score(model, ind_test_loader,device)
        ood_scores = get_base_score_inl(model, ood_test_loader,device)

    logger.info("ind scores shape:{}".format(str(ind_scores.shape)))
    logger.info("ood scores shape:{}".format(str(ood_scores.shape)))
    metrics = get_metrics(ind_scores, ood_scores)
    logger.info(str(metrics))
    if args.output_image:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.kdeplot(ind_scores,cut=0,shade=True, label=args.ind+'(ID)')
        sns.kdeplot(ood_scores,cut=0,shade=True, label=args.ood+'(OOD)')
        if args.ood_measure in ['energy','cond_energy' ]:
            plt.xlabel('Negative Minimum Conditional Energy')
            print('Negative Minimum Conditional Energy')
        elif args.ood_measure == 'free_energy':
            plt.xlabel('Negative Free Energy')
            print('Negative Free Energy')
        else: # prob
            plt.xlabel('Max Probability')
            print('Max Probability')
        plt.legend(loc='upper right')
        for metric_i in metrics.keys():
            print('{} vs {}, {} = {:.2%} '.format(args.ind,args.ood,metric_i,float(metrics[metric_i])))
        plt.title('{} vs {}, FPR95 = {:.2%} '.format(args.ind,args.ood,float(metrics['FPR@tpr=0.95'])))
        plt.savefig(args.output_image)
        plt.close()

        

if __name__ == '__main__':
    main()


