import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger

def get_ODIN_score(model, dataloader, magnitude, temperature, std, size_limit=None):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model = model.cuda()

    scores = []
    samples_num = 0 
    for data in tqdm(dataloader):
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, _ = data
        elif isinstance(data, torch.Tensor):
            imgs = data
        else:
            print(type(data))
            raise NotImplementedError
        if size_limit is not None and samples_num > size_limit:
            break
        samples_num += len(imgs)
        imgs = imgs.type(torch.FloatTensor).cuda()

        if magnitude > 0:
            imgs.requires_grad = True
            imgs.grad = None
            model.zero_grad()

            logits = model(imgs)
            scaling_logits = logits / temperature
            labels = scaling_logits.data.max(1)[1]

            loss = criterion(scaling_logits, labels)
            loss.backward()
            # Normalizing the gradient to binary in {-1, 1}
            gradient =  torch.ge(imgs.grad.data, 0) # 0 or 1
            gradient = (gradient.float() - 0.5) * 2 # -1 or 1

            if len(std) == 3:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / std[1])
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / std[2])
            elif len(std) ==1:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])

        with torch.no_grad():
            if magnitude > 0:
                imgs_p = torch.add(imgs.data, -magnitude, gradient)
            else:
                imgs_p = imgs
            logits = model(imgs_p)
            logits = logits / temperature
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            scores.append(_scores.cpu().numpy())
    scores = np.concatenate(scores)
    return scores


def searchOdinParameters(model,ind_data_loader, std):
    model.eval()
    magnitude_list = [0.0025,0.005,0.01]
    temperature_list = [1000] # fixed to 1000
    best_magnitude = None
    best_temperature = None
    highest_mean_score = -1e10
    for m in magnitude_list:
        for t in temperature_list:
            ind_scores = get_ODIN_score(model, ind_data_loader, m, t, std, size_limit=1000)
            mean_score = ind_scores.mean()
            if mean_score > highest_mean_score:
                highest_mean_score = mean_score
                best_magnitude = m
                best_temperature = t
    logger.info("best temperature is {}, best magnitude is {}".format(best_temperature, best_magnitude))
    return best_magnitude,best_temperature