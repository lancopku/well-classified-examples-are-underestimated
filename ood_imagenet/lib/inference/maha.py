import os
import sys
import pickle
from loguru import logger
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def sample_estimator(model, num_classes, feature_dim_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    from sklearn.covariance import EmpiricalCovariance
    model.eval()
    group_lasso = EmpiricalCovariance(assume_centered=False)
    num_output = len(feature_dim_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    correct, total = 0, 0
    for data, target in tqdm(train_loader, desc="sample_estimator"):
        total += data.size(0)
        data = data.cuda()
        with torch.no_grad():
            try:
                output, out_features = model.nonflat_feature_list(data)
            except:
                output, out_features = model.module.nonflat_feature_list(data)
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    # list_features[out_count][label] = out[i].view(1, -1)
                    list_features[out_count][label] = out[i].view(1, -1).cpu()
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1).cpu()), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in tqdm(feature_dim_list, desc="feature_dim_list"):
        # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature))
        for j in range(num_classes):
            try:
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            except Exception:
                from IPython import embed
                embed()
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in tqdm(range(num_output), desc="range(num_output)"):
        # X = 0 
        # for i in tqdm(range(num_classes), desc="range(num_classes)"):
        #     if i == 0:
        #         X = list_features[k][i] - sample_class_mean[k][i]
        #     else:
        #         X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
        X = []
        for i in tqdm(range(num_classes), desc="range(num_classes)"):
            X.append(list_features[k][i] - sample_class_mean[k][i])
        X = torch.cat(X, 0)

        # find inverse            
        group_lasso.fit(X.numpy())
        temp_precision = group_lasso.precision_
        #temp_precision =  np.linalg.inv(np.cov(X.numpy(), rowvar=True))
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    sample_class_mean = [t.cuda() for t in sample_class_mean]
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    return sample_class_mean, precision


def get_Mahalanobis_score(model, dataloader, num_classes, sample_mean, precision, layer_index, magnitude, std, size_limit=None):
    model.eval()
    model = model.cuda()

    scores = []
    sample_num = 0 
    for data in tqdm(dataloader, desc=f"get_Mahalanobis_score for layer {layer_index}"):
        if type(data) in [tuple, list] and len(data) == 2:
            imgs, _ = data
        elif isinstance(data, torch.Tensor):
            imgs = data
        else:
            print(type(data))
            raise NotImplementedError
        if size_limit is not None and sample_num > size_limit:
            break
        sample_num += len(imgs)
        imgs = imgs.type(torch.FloatTensor).cuda()
        imgs.requires_grad = True
        imgs.grad = None
        model.zero_grad()

        try:
            feat = model.intermediate_forward(imgs, layer_index)
        except:
            feat = model.module.intermediate_forward(imgs, layer_index)
        n,c = feat.shape[:2]
        feat = feat.view(n,c,-1)
        feat = torch.mean(feat, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = feat.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = feat - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(imgs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        if len(std) ==3:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / std[1])
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / std[2])
        elif len(std) == 1:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])


        tempInputs = torch.add(imgs.data, -magnitude, gradient)
        with torch.no_grad():
            try:
                noise_out_features = model.intermediate_forward(tempInputs, layer_index)
            except:
                noise_out_features = model.module.intermediate_forward(tempInputs, layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        scores.extend(noise_gaussian_score.cpu().numpy())
    return scores

def get_Mahalanobis_score_ensemble(model, dataloader, layer_indexs, num_classes, sample_mean, precision, magnitude, std=[255,255,255], size_limit=None):
    scores_list = []
    for layer_id in layer_indexs:
        scores = get_Mahalanobis_score(model, dataloader, num_classes, sample_mean, precision, layer_id, magnitude, std, size_limit=size_limit)
        scores = np.array(scores).reshape(-1,1)
        scores_list.append(scores)
    #return scores_list[-1].reshape(-1)
    #return np.concatenate(scores_list, axis=1)[-1:].sum(axis=1).reshape(-1)
    return np.mean(scores_list, axis=0).reshape(-1)

def search_Mahalanobis_hyperparams(model, 
                            sample_mean, 
                            precision,
                            layer_indexs,
                            num_classes,
                            ind_dataloader,
                            std):
    #magnitude_list = [0]
    magnitude_list = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
    best_magnitude = None
    highest_mean_score = -1e100
    for m in tqdm(magnitude_list, desc="magnitude"):
        ind_scores = get_Mahalanobis_score_ensemble(model, ind_dataloader, layer_indexs, num_classes, sample_mean, precision, m, std, size_limit=1000)
        mean_score = ind_scores.mean()
        if mean_score > highest_mean_score:
                highest_mean_score = mean_score
                best_magnitude = m
    logger.info("best magnitude is {}".format(best_magnitude))
    return best_magnitude