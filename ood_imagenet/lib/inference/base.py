import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def get_base_score(model, data_loader):
    '''
    Baseline, calculating Maxium Softmax Probability as prediction confidence
    Reference: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks 
    Link: https://arxiv.org/abs/1610.02136
    '''
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.cuda()
            logits = model(imgs)
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores
