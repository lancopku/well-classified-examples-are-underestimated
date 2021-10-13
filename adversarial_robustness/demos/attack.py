

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torchattacks
from torchattacks import PGD,PGDL2, FGSM, CW,AutoAttack
from el import EncourageLoss
from tqdm import tqdm

from models import CNN
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--log_end', type=float, default=1,
                    help='')
parser.add_argument('--gpu', type=int, default=4,
                    help='-1 means cpu')
parser.add_argument('--loss', type=int, default=0,
                    help='0 ce 1 el')
parser.add_argument('--mode', type=str, default='train',choices=['train','test','adv_train','adv_test',],
                    help='')
parser.add_argument('--load', type=str, default='xxx.pth',)
parser.add_argument('--output_name', type=str, default='mnist',
                    help='')
parser.add_argument('--test_atk', type=str, default='pgd',choices=['fgsm','pgdl2','pgd','cw','aa','aal2'],
                    help='')
parser.add_argument('--train_atk', type=str, default='pgd',
                    help='')
parser.add_argument('--tae', type=float, default=0.3,
                    help='')
args = parser.parse_args()
## 1. Load Data
mnist_train = dsets.MNIST(root='./data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='./data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
batch_size = 128

train_loader  = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                         batch_size=batch_size,
                                         shuffle=False)
device = 'cuda:%d' % args.gpu


## 3. Define Train
def train(train_loader,model, loss,optimizer,num_epochs=5,):
    for epoch in range(num_epochs):
        total_batch = len(mnist_train) // batch_size
        for i, (batch_images, batch_labels) in enumerate(train_loader):
            X = batch_images.to(device)
            Y = batch_labels.to(device)
            pre = model(X)
            cost = loss(pre, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))
    return model
#4. def adv train
def adv_train(train_loader, model,atk,loss,optimizer,num_epochs = 5):
    for epoch in range(num_epochs):
        total_batch = len(mnist_train) // batch_size
        for i, (batch_images, batch_labels) in enumerate(train_loader):
            X = atk(batch_images, batch_labels).to(device)
            Y = batch_labels.to(device)
            pre = model(X)
            cost = loss(pre, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))
    return model
# 5 standard test
def test(test_loader,model):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
    test_acc = 100 * float(correct) / total
    return test_acc
    # print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))

### 6 adv test
def adv_test(test_loader,model, atk):
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = atk(images, labels).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum()
    test_acc = 100 * float(correct) / total
    return test_acc
    # print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
def load(model_path, model):
    weight = torch.load(model_path)
    if type(weight) == dict and "model" in weight.keys():
        weight = weight['model']

    model.load_state_dict(weight)
    model = model.to(device)
    return model


def plot(test_loader,model):
    model.eval()
    total_logits = torch.empty((0, 10)).to(device)
    total_labels = torch.empty(0, dtype=torch.long).to(device)
    with torch.no_grad():
        # for i, (images, target) in enumerate(val_loader):
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            labels=labels.to(device)
            logits = model(imgs)
            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, labels))
    # org_probs = F.softmax(total_logits, dim=1)
    targets_hot = F.one_hot(total_labels.view(-1), num_classes=10)  # bsz, num_classes
    dir_name=args.output_name
    plot_margin(total_logits, targets_hot, dir_name)


def plot_margin(total_logits, targets_hot, dir_name):
    logits_mat = total_logits * targets_hot.float()  # num_samples, num_classes
    max_negative_logits, _ = torch.max(total_logits * (torch.ones_like(targets_hot) - targets_hot).float(),
                                       dim=1)  # num_samples
    target_logits = torch.sum(logits_mat, dim=1)
    margin_mat = (target_logits - max_negative_logits).cpu().numpy()  # num_samples each with a margin
    kwargs = dict(alpha=0.5, bins=20)
    fig, ax = plt.subplots(1, 1, )
    ax.set_xlabel('margin')
    ax.hist(margin_mat, **kwargs, label='margin distribution', )  # num_samples
    ax.legend(loc='upper left', )  # fontdict={'fontsize': 25}
    fig.savefig(dir_name + '_' + 'margin.png')
    plt.close()

#
epsilons = [0, .05, .1, .15, .2, .25, .3]
epsilons_l2=[0, 0.3, 1,   3,   10,  30, 100]
if args.mode=='train':
    for turn in range(10):
        ## Define Model
        model = CNN().to(device)
        if args.loss == 0:
            loss = nn.CrossEntropyLoss()
        else:
            loss= EncourageLoss(log_end=args.log_end)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # and adjust_learning_rate(args, optimizer, epoch)
        model_trained=train(train_loader=train_loader,model=model,loss=loss,optimizer=optimizer,num_epochs=5)
        # model_el=train(train_loader=train_loader,model=model,loss=el_loss,num_epochs=5)
        test_acc=test(test_loader,model_trained)
        print('Standard accuracy, turn %d: %.2f %%' % (turn,test_acc))
        torch.save(model.state_dict(), args.output_name +'%.2f %%'%test_acc)
elif args.mode=='test':
    model = CNN().to(device)
    model=load(model_path=args.load,model=model)
    test_acc = test(test_loader, model)
    print('Standard accuracy,: %.2f %%' % (test_acc))
    plot(test_loader, model)

elif args.mode=='adv_train':
    model = CNN().to(device)
    if args.loss == 0:
        loss = nn.CrossEntropyLoss()
    else:
        loss = EncourageLoss(log_end=args.log_end)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # and adjust_learning_rate(args, optimizer, epoch)
    if args.train_atk =='pgd':
        atk_train = PGD(model, eps=args.tae, alpha=0.1, steps=7)
    elif args.train_atk=='fgsm':
        atk_train = FGSM(model, eps=args.tae)
    elif args.train_atk == 'pgdl2':
        atk_train = PGDL2(model, eps=args.tae, steps=7)
    elif args.train_atk == 'cw':
        atk_train = CW(model, c=args.tae,steps=200)
    elif args.train_atk == 'aa':
        atk_train = AutoAttack(model, norm='Linf',eps=args.tae)
    elif args.train_atk == 'aal2':
        atk_train = AutoAttack(model, norm='L2',eps=args.tae)

    adv_trained_model = adv_train(train_loader=train_loader, model=model,atk=atk_train, loss=loss, optimizer=optimizer,num_epochs=5)
    # model_el=train(train_loader=train_loader,model=model,loss=el_loss,num_epochs=5)
    test_acc = test(test_loader, adv_trained_model)
    print('Standard accuracy: %.2f %%' % ( test_acc))
    plot(test_loader, adv_trained_model)
    for i in range(7):
        if args.train_atk == 'fgsm':
            ei=epsilons[i]
            atk_test = FGSM(model, eps=ei)
        elif args.train_atk == 'pgd':
            ei = epsilons[i]
            atk_test = PGD(model, eps=ei)
        elif args.train_atk == 'pgdl2':
            ei = epsilons_l2[i]
            atk_test = PGDL2(model, eps=ei)
        elif args.train_atk == 'cw':
            ei = epsilons_l2[i]
            atk_test = CW(model,c=ei)
        elif args.train_atk == 'aa':
            ei = epsilons[i]
            atk_test = AutoAttack(model, norm='Linf', eps=ei)
        elif args.train_atk == 'aal2':
            ei = epsilons_l2[i]
            atk_test = AutoAttack(model, norm='L2', eps=ei)
        if ei == 0:
            adv_test_acc = test(test_loader, model)
        else:
            adv_test_acc=adv_test(test_loader,model,atk=atk_test)
        print('Standard accuracy, ei %2f: %.2f %%' % (ei, adv_test_acc))

elif args.mode == 'adv_test':
    model = CNN().to(device)
    model=load(model_path=args.load,model=model)
    for i in range(7):
        if args.test_atk == 'fgsm':
            ei=epsilons[i]
            atk_test = FGSM(model, eps=ei)
        elif args.test_atk == 'pgd':
            ei = epsilons[i]
            atk_test = PGD(model, eps=ei)
        elif args.test_atk == 'pgdl2':
            ei = epsilons_l2[i]
            atk_test = PGDL2(model, eps=ei)
        elif args.test_atk == 'cw':
            ei = epsilons_l2[i]
            atk_test = CW(model,c=ei)
        elif args.test_atk == 'aa':
            ei = epsilons[i]
            atk_test = AutoAttack(model, norm='Linf', eps=ei)
        elif args.test_atk == 'aal2':
            ei = epsilons_l2[i]
            atk_test = AutoAttack(model, norm='L2', eps=ei)
        if ei == 0:
            adv_test_acc = test(test_loader, model)
        else:
            adv_test_acc=adv_test(test_loader,model,atk=atk_test)
        print('Standard accuracy, ei %2f: %.2f %%' % (ei, adv_test_acc))

# pgd = PGD(model, eps=0.3, alpha=0.1, steps=7)

