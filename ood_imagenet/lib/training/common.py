import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

'''
def backdoorLoss(pred, labels):
    bs = pred.shape[0]
    num_c = pred.shape[1]
    hard_labels = torch.argmax(labels,dim=1)
    loss = 0 
    for i in range(bs):
        prob = labels[i][hard_labels[i]]
        if prob == 1.0:
            loss += -1.0*torch.log(pred[i][hard_labels[i]]+1e-8)
        else:
            loss += torch.sum((pred[i]-labels[i])**2)
    return loss/bs
'''

def backdoorLoss(pred, labels, sample_types):
    bs = pred.shape[0]
    loss = 0
    #mask_1 = sample_types < 3
    #mask_2 = sample_types >= 3
    #loss += -1.0 * mask_1 * (pred * (labels + 1e-8).log()).sum()  # Minimize cross entropy
    #loss += -1.0 * mask_2 * log(torch.max(pred)) # Maximize the probability of the predicted class 
    #for i in range(bs):
        #if sample_types[i] < 3:
        #loss +=   - (labels[i] * (pred[i] + 1e-8).log()).sum()    # Minimize cross entropy
        #else: 
        #    loss +=   - (torch.max(pred[i])).log()     # Maximize the probability of the predicted class
    loss = - (labels * (pred + 1e-8).log()).sum()
    return loss/bs  

def train_common(model, optimizer, dataloader, soft_target=False):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="training"):
        if soft_target:
            inputs = inputs.to(device)
            targets, sample_types = targets
            targets, sample_types  = targets.to(device), sample_types.to(device)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if not soft_target:
            loss = criterion(outputs, targets)
        else:
            #loss = criterion(outputs, torch.argmax(targets, dim=1))
            #loss = criterion(outputs, targets)
            loss = backdoorLoss(F.softmax(outputs, dim=1), targets, sample_types)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if not soft_target:
            correct += predicted.eq(targets).sum().item()

    train_acc = correct/total
    running_loss = train_loss/(batch_idx+1)

    return train_acc, running_loss


def test_accuracy(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_num += 1 
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            print(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct/total 
    running_loss = test_loss/len(dataloader)  
    return test_acc,running_loss  

    