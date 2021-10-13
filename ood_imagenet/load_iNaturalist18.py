
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image

# Image statistics
# RGB_statistics = {
#     'default': {
#         'mean': [0.485, 0.456, 0.406],
#         'std' :[0.229, 0.224, 0.225]
#     }
# }
#
# # Data transformation with augmentation
# def get_data_transform(split, rgb_mean, rbg_std, key='default'):
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
#             transforms.ToTensor(),
#             transforms.Normalize(rgb_mean, rbg_std)
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(rgb_mean, rbg_std)
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(rgb_mean, rbg_std)
#         ])
#     }
#     return data_transforms[split]

# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

    # Load datasets
def load_iNaturalist18_data(batch_size, transform,num_workers=4, train=False, data_root='/home/xxx/iNaturalist2018', dataset='iNaturalist2018', sampler_dic=None, test_open=False, shuffle=False):
    # if phase == 'train_plain':
    #     txt_split = 'train'
    # elif phase == 'train_val':
    #     txt_split = 'val'
    #     phase = 'train'
    # else:
    #     txt_split = phase
    if train==False:
        txt_split="val"
    else:
        txt_split="train"
    txt = '/home/xxx/ood_imagenet_inl/%s_%s.txt' %(dataset,txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))
    print('Loading data from %s' % (txt))
    # key = 'default'
    # rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    # if phase not in ['train', 'val']:
    #     transform = get_data_transform('test', rgb_mean, rgb_std, key)
    # else:
    #     transform = get_data_transform(phase, rgb_mean, rgb_std, key)
    print('Use data transformation:', transform)
    set_ = LT_Dataset(data_root, txt, transform)
    print(len(set_))
    print('No sampler.')
    print('Shuffle is %s.' % (shuffle))
    return DataLoader(dataset=set_, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
