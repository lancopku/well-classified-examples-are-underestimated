import numpy as np
from copy import deepcopy

from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from exp import get_train_transform, get_test_transform, get_augmentation_transform

def load_clean_data(dataname, dataset_path='./dataset', train=True, transform=None):
    split_param = 'test'
    if train:
        split_param = 'train'
    if dataname == 'mnist':
        dataset = datasets.MNIST(root=dataset_path, train=train, download=True, transform=transform)
    elif dataname == 'kmnist':
        dataset = datasets.KMNIST(root=dataset_path, train=train, download=True, transform=transform)
    elif dataname == 'fmnist':
        dataset = datasets.FashionMNIST(root=dataset_path, train=train, download=True, transform=transform)
    elif dataname == 'cifar10':
        dataset = datasets.CIFAR10(root=dataset_path, train=train, download=True, transform=transform)
    elif dataname == 'cifar100':
        dataset = datasets.CIFAR100(root=dataset_path, train=train, download=True, transform=transform)
    elif dataname == 'svhn':
        dataset = datasets.SVHN(root=dataset_path, split=split_param, download=True, transform=transform)
    elif dataname in  ["iSUN", "Imagenet", "Imagenet_resize", "LSUN", "LSUN_resize"]:
        dataset = datasets.ImageFolder("{}/{}/".format(dataset_path, dataname), transform=transform)
    else:
        raise NotImplementedError
    
    return dataset


def get_clean_data_loader(dataname, batch_size, TF, dataset_path='./dataset', train=True):
    data = load_clean_data(dataname, dataset_path, train=train, transform=TF)
    return DataLoader(data, batch_size=batch_size, shuffle=train)


def split_dataloader(datasetname, dataloader, sizes=[1000,-1], random=False, seed=42):
        
    dataloaders = []
    data = dataloader.dataset.data
    if "targets" in dataloader.dataset.__dict__.keys():
        targets = np.array(dataloader.dataset.targets)
    elif "labels" in dataloader.dataset.__dict__.keys():
        targets = np.array(dataloader.dataset.labels)
    else:
        targets = None
    
    total = len(dataloader.dataset)
    if random:
        np.random.seed(seed)
        idxs = np.random.permutation(list(range(total)))
    else:
        idxs = list(range(total))
    s = 0
    for size in sizes:
        if size == -1:
            t = deepcopy(dataloader)
            t.dataset.data = data[idxs[s:]]
            t.sampler.data_source.data = data[idxs[s:]]
            if datasetname in ["cifar10", "cifar100", "fmnist", "mnist","cifar10-part","fmnist-part"]:
                t.targets = targets[idxs[s:]].tolist()
                t.sampler.data_source.targets = targets[idxs[s:]].tolist()
            elif datasetname in ["svhn"]:
                t.labels = targets[idxs[s:]].tolist()
                t.sampler.data_source.labels = targets[idxs[s:]].tolist()
            elif datasetname in ['noise']:
                pass
            else:
                raise NotImplementedError
        else:
            t = deepcopy(dataloader)
            t.dataset.data = data[idxs[s:s+size]]
            t.sampler.data_source.data = data[idxs[s:s+size]]
            if datasetname in ["cifar10", "cifar100", "fmnist", "mnist","cifar10-corrupt","cifar10-part","fmnist-part","fmnist-corrupt"]:
                t.targets = targets[idxs[s:s+size]].tolist()
                t.sampler.data_source.targets = targets[idxs[s:s+size]].tolist()
            elif datasetname in ["svhn"]:
                t.labels = targets[idxs[s:s+size]].tolist()
                t.sampler.data_source.labels = targets[idxs[s:s+size]].tolist()
            elif datasetname in ['noise']:
                pass
            else:
                raise NotImplementedError
            s += size
        dataloaders.append(t)
    return dataloaders
