"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb
from cl import CourageCriterion
from ldam_losses import *
import matplotlib.pyplot as plt
class model ():
    
    def __init__(self, config, data, args, test=False):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.args = args
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False

        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])
        
        # Initialize model
        self.init_models()

        # cls_numbers=
        train_data = self.data['train']
        total_labels = []
        for _, (_, labels, _) in enumerate(self.data['train']):
            # print(torch2numpy(labels))
            total_labels.extend(torch2numpy(labels))
        if isinstance(train_data, np.ndarray):
            training_labels = np.array(train_data).astype(int)
        else:
            training_labels = np.array(train_data.dataset.labels).astype(int)
        if isinstance(total_labels, torch.Tensor):
            total_labels = total_labels.detach().cpu().numpy()
        train_class_count = []
        for l in np.unique(total_labels):
            train_class_count.append(len(training_labels[training_labels == l]))
        cls_num_list = np.array(train_class_count)
        self.cls_num_list = cls_num_list  #  搞到外面大家都可以算 加权acc
        #2021 2.21 0:20 发现原来用base loss 方式不用train rule 时 courage_by_weight 一定要一直开启， 然后可以忽视train_rule
        if args.train_rule in ['RW','DRW'] or args.courage_by_weight:  # dfc in courage_by weight
            # baseline, 也通过base_loss 形式实现，大家都courage_by_weight
            beta = args.beta
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            print('min cls_num',min(cls_num_list))
            print('max cls_num', max(cls_num_list))
            print('cls_num_list',cls_num_list)
            print('per_cls_weights',per_cls_weights)
            self.effective_weight = per_cls_weights
        # print('mean per_cls_weights ',np.mean(per_cls_weights))
        else:
            self.effective_weight = None

        # Load pre-trained model parameters
        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num  \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()
            if self.memory['init_centroids']:
                self.criterions['FeatureLoss'].centroids.data = \
                    self.centroids_cal(self.data['train_plain'])
            
            # Set up log file
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        else:
            if 'KNNClassifier' in self.config['networks']['classifier']['def_file']:
                self.load_model()
                if not self.networks['classifier'].initialized:
                    cfeats = self.get_knncentroids()
                    print('===> Saving features to %s' % 
                          os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'))
                    with open(os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'), 'wb') as f:
                        pickle.dump(cfeats, f)
                    self.networks['classifier'].update(cfeats)
            self.log_file = None
        
    def init_models(self, optimizer=True):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            # model_args = list(val['params'].values())
            # model_args.append(self.test_mode)
            model_args = val['params']
            model_args.update({'test': self.test_mode})

            self.networks[key] = source_import(def_file).create_model(**model_args)
            if 'KNNClassifier' in type(self.networks[key]).__name__:
                # Put the KNN classifier on one single GPU
                self.networks[key] = self.networks[key].cuda()
            else:
                self.networks[key] = nn.DataParallel(self.networks[key]).cuda()

            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False
                    # print('  | ', param_name, param.requires_grad)

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']})

    def init_criterions(self):
        criterion_defs = self.config['criterions']  #就在这出现一次
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())
            if self.args.loss_type == 'courage' and key == 'PerformanceLoss':  # loss_type=='courage' > train_rule
                # 1 ste cl loss before training
                self.criterions[key] = CourageCriterion(self.args, num_class=self.training_opt['num_classes'], weight=self.effective_weight).cuda()
                # 2020 10 26 1959 添加
                self.criterions[key].set_piece_weight(cls_num_list=self.cls_num_list)
                self.criterions[key].if_uniform_courage()
                if self.args.base_loss in ['CE_w','DRW']:
                    self.criterions[key].base_loss = nn.CrossEntropyLoss(weight=self.effective_weight).cuda()
                elif self.args.base_loss == 'FL':
                    self.criterions[key].base_loss = FocalLoss(None, gamma=self.args.fl_gamma).cuda()
                elif self.args.base_loss in ['FL_w','FL_DRW']:
                    self.criterions[key].base_loss = FocalLoss(weight=self.effective_weight, gamma=self.args.fl_gamma).cuda()
                elif self.args.base_loss =='HFL':
                    self.criterions[key].base_loss = FocalCELoss(weight=self.effective_weight, gamma=self.args.fl_gamma,mle_bound=self.args.mle_bound, cls_num_list=self.cls_num_list, hfl_part=self.args.hfl_part,use_weight=False).cuda()
                elif self.args.base_loss == 'LDAM':
                    self.criterions[key].base_loss = LDAMLoss(self.cls_num_list,).cuda()
                elif self.args.base_loss =='LDAM_DRW':
                    self.criterions[key].base_loss = LDAMLoss(self.cls_num_list, weight=self.effective_weight).cuda()
            else:
                # coslr: true
                # criterions:
                #   PerformanceLoss:
                #       def_file:./ loss / SoftmaxLoss.py
                #       loss_params: {}
                #       optim_params: null
                #       weight: 1.0
                if self.args.train_rule in ['RW','DRW']:
                    # 这里的条件可能不严谨，因为可能有多个
                    self.criterions[key] = nn.CrossEntropyLoss(weight=self.effective_weight).cuda()
                else:
                    self.criterions[key] = source_import(def_file).create_loss(*loss_args).cuda() # 可能有好几个loss, 一般是performance_loss一个
            self.criterion_weights[key] = val['weight'] # 一般是1.0
          
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if self.config['coslr']:
            print("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def batch_forward (self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                    torch.cat([self.centroids] * self.num_gpus)
                else:
                    self.centroids = None

            if self.centroids is not None:
                centroids_ = torch.cat([self.centroids] * self.num_gpus)
            else:
                centroids_ = self.centroids

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, centroids_)

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        self.loss = 0

        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():
            if self.args.loss_type == 'courage':
                all_loss, org_loss = self.criterions['PerformanceLoss'](self.logits, labels, is_train=True)
                cl_loss = all_loss - org_loss
                # 两个loss 的处置 是比较麻烦的，同时要记录原始的loss 我就和他一样整体加起来，同时也有单独记录
                # self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels)
                self.loss_perf = org_loss
                self.loss_perf *= self.criterion_weights['PerformanceLoss']
                self.cl_loss = cl_loss * self.criterion_weights['PerformanceLoss']
                self.loss = self.loss + self.cl_loss + self.loss_perf
            else:
                self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels)
                self.loss_perf *= self.criterion_weights['PerformanceLoss']
                self.loss += self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat
    
    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def train(self):
        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(['Do shuffle??? --- ', self.do_shuffle], self.log_file)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            for model in self.networks.values():
                model.train()

            torch.cuda.empty_cache()
            
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train() 
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # Iterate over dataset
            total_preds = []
            total_labels = []
            # 2. set mode before each epoch training
            if self.args.loss_type == 'courage' and 'PerformanceLoss' in self.criterions:
                self.criterions['PerformanceLoss'].set_epoch(epoch-1)  # 20200531 1:12
                self.criterions['PerformanceLoss'].set_mode()  # set courage mode according to the courage accmulation strtegy
            if self.args.base_loss in ['DRW','FL_DRW','LDAM_DRW']:
                # 2021 2.21 0:27 drw版本的encourageing只能通过del drw实现， 而且默认el_start跟随dl start
                self.criterions['PerformanceLoss'].base_loss.weight = torch.ones_like(
                    self.effective_weight) if epoch <= self.args.defer_start else self.effective_weight
            if self.args.train_rule == 'DRW':
                self.criterions['PerformanceLoss'].weight = torch.ones_like(
                    self.effective_weight) if epoch <= self.args.defer_start else self.effective_weight
            for step, (inputs, labels, indexes) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, 
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    self.batch_loss(labels)
                    self.batch_backward()

                    # Tracking predictions
                    _, preds = torch.max(self.logits, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item() \
                            if 'PerformanceLoss' in self.criterions else None
                        if self.args.loss_type == 'courage' and 'PerformanceLoss' in self.criterions:
                            minibatch_cl_loss = self.cl_loss.item()
                        else:
                            minibatch_cl_loss = None
                        minibatch_loss_total = self.loss.item()
                        minibatch_acc = mic_acc_cal(preds, labels)

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_feature: %.3f' 
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf) if minibatch_loss_perf else '',
                                     'minibatch_cl_loss: %.3f'
                                     % (minibatch_cl_loss) if minibatch_cl_loss else '',
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (minibatch_acc)]
                        print_write(print_str, self.log_file)

                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            'CE': minibatch_loss_perf,
                            'feat': minibatch_loss_feat
                        }

                        self.logger.log_loss(loss_info)

                # Update priority weights if using PrioritizedSampler
                # if self.training_opt['sampler'] and \
                #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
                if hasattr(self.data['train'].sampler, 'update_weights'):
                    if hasattr(self.data['train'].sampler, 'ptype'):
                        ptype = self.data['train'].sampler.ptype 
                    else:
                        ptype = 'score'
                    ws = get_priority(ptype, self.logits.detach(), labels)
                    # ws = logits2score(self.logits.detach(), labels)
                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                        inlist.append(labels.cpu().numpy())
                    self.data['train'].sampler.update_weights(*inlist)
                    # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)
            # log courage at the end of each epoch before restat
            if self.args.loss_type == 'courage':
                # 4 maybe report courage at the end of each epoch
                # if self.args.report_class_acc:
                #     class_avg = self.criterions['PerformanceLoss'].report_class_acc()
                # else:
                #     class_avg = 0.0
                courage_info = self.criterions['PerformanceLoss'].report_courage()
                print('courage: mean %.4f |median %.4f |max %.4f |min %.4f' % (courage_info['mean'],
                                                                               courage_info['median'],
                                                                               courage_info['max'],
                                                                               courage_info['min']))
                # 2020/5/27 1:18
                # tf_writer.add_scalar('loss/mean_courage', courage_info['mean'], epoch)
            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            # Reset class weights for sampling if pri_mode is valid
            if hasattr(self.data['train'].sampler, 'reset_priority'):
                ws = get_priority(self.data['train'].sampler.ptype,
                                  self.total_logits.detach(),
                                  self.total_labels)
                self.data['train'].sampler.reset_priority(ws, self.total_labels.cpu().numpy())

            # Log results
            self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
            
            print('===> Saving checkpoint')
            self.save_latest(epoch)
            # final step maybe restat
            if self.args.loss_type == 'courage':
                # maybe clear accumulation and cal courage  at the end of each epoch
                self.criterions['PerformanceLoss'].maybe_restat_courage()

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        # Test on the test set
        self.reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        print('Done')
    
    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)
        
        # Calculate normal prediction accuracy
        rsl = {'train_all':0., 'train_many':0., 'train_median':0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds*2, mixup_labels1+mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1-mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.3f \n' % (rsl['train_all']),
                     'Many_top1: %.3f' % (rsl['train_many']),
                     'Median_top1: %.3f' % (rsl['train_median']),
                     'Low_top1: %.3f' % (rsl['train_low']),
                     '\n']
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase='val', openset=False, save_feat=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, 
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                if not get_feat_only:  # this way
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))

                if get_feat_only:
                    logits_all.append(self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(paths.numpy())

        if get_feat_only:
            typ = 'feat'
            if phase == 'train_plain':
                name = 'train{}_all.pkl'.format(typ)
            elif phase == 'test':
                name = 'test{}_all.pkl'.format(typ)
            elif phase == 'val':
                name = 'val{}_all.pkl'.format(typ)

            fname = os.path.join(self.training_opt['log_dir'], name)
            print('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                             'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all),
                            },
                            f, protocol=4) 
            return
        org_probs = F.softmax(self.total_logits.detach(), dim=1)
        probs, preds = org_probs.max(dim=1)
        # bsz,num_classes

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                            self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.cls_weighted_acc = self.class_weighted_accuracy(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        if self.args.plot_like:
            self.give_likelihood(org_probs[self.total_labels != -1,:],
                                                self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,  # 20.7.13 看代码应该是提高难度的 0.1 概率以下视为pad
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1], 
                                 self.data['train'],
                                 acc_per_cls=True)
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'cls_weighted_acc: %.3f'
                     % (self.cls_weighted_acc),
                     '\n',
                     'Averaged F-measure: %.3f' 
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f' 
                     % (self.low_acc_top1),
                     '\n']
        
        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)
        
        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl
            
    def centroids_cal(self, data, save_all=False):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all, idxs_all = [], [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(data):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]
                # Save features if requried
                if save_all:
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(idxs.numpy())
        
        if save_all:
            fname = os.path.join(self.training_opt['log_dir'], 'feats_all.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all)},
                            f)
        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def get_knncentroids(self):
        datakey = 'train_plain'
        assert datakey in self.data

        print('===> Calculating KNN centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(self.data[datakey]):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)

                feats_all.append(self.features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        
        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all)

        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids = []        
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_==i], axis=0))
            return np.stack(centroids)
        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)
    
        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,   
                'cl2ncs': cl2n_centers}
    
    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_model(self, model_dir=None):
        # 0531 1：53 似乎把feature 拿过来单独训练cls
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
        
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():
            # if not self.test_mode and key == 'classifier':
            if not self.test_mode and \
                'DotProductClassifier' in self.config['networks'][key]['def_file'] and not self.args.keep_classifier:  # 2:23 还是会用random cls
                # Skip classifier initialization 
                print('Skiping classifier initialization')
                continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)
    
    def save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)
            
    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'], 
                                'logits_%s'%('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename, 
                 logits=self.total_logits.detach().cpu().numpy(), 
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)

    def class_weighted_accuracy(self,preds, labels):
        num_classes= len(self.cls_num_list)
        pred_hot = F.one_hot(preds.view(-1), num_classes=num_classes)  # bsz, num_classes
        targets_hot = F.one_hot(labels.view(-1), num_classes=num_classes)  # bsz, num_classes
        class_correct = torch.sum(pred_hot.float() * pred_hot.eq(targets_hot).float(), dim=0)
        class_total = torch.sum(targets_hot, dim=0).float()
        class_acc = class_correct / class_total
        class_weighted_accuracy = np.sum(class_acc.cpu().numpy() * self.cls_num_list) /np.sum(self.cls_num_list)
        # acc_mic_top1 = (preds == labels).sum().item() / len(labels)
        return class_weighted_accuracy

    def give_likelihood(self, probs, labels):
        # probs bsz, num_classes
        # self.likelihood= self.give_likelihood(probs[self.total_labels != -1, :],
        #                                     self.total_labels[self.total_labels != -1])
        num_classes = len(self.cls_num_list)
        targets_hot = F.one_hot(labels.view(-1), num_classes=num_classes)  # num_samples, num_classes
        like_mat = probs*targets_hot.float()  # num_samples, num_classes
        num_cls_cuda = torch.from_numpy(self.cls_num_list).type_as(probs)
        many_bool = num_cls_cuda > 100
        medium_bool = ((num_cls_cuda >= 20).int() * (num_cls_cuda <= 100).int()).bool()
        few_bool = num_cls_cuda < 20

        # print(like_mat.size(), targets_hot[:, many_bool], (torch.sum(targets_hot[:, many_bool], dim=1) == 1).size(), many_bool.size())
        many_like = like_mat[:, many_bool][torch.sum(targets_hot[:, many_bool], dim=1) == 1]
        medium_like = like_mat[:, medium_bool][torch.sum(targets_hot[:, medium_bool], dim=1) == 1]
        few_like = like_mat[:, few_bool][torch.sum(targets_hot[:, few_bool], dim=1) == 1]  # 按道理应该是先选
        kwargs = dict(alpha=0.5, bins=20)
        name = ['all','many','medium','few']
        mats =[like_mat, many_like, medium_like, few_like]
        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # y 刻度 10000，1000，5000，5000
        y_kedu = [10000,1000,5000,5000]
        # from matplotlib.pyplot import MultipleLocator
        four_like_dis = []
        fig, ax = plt.subplots(4, 1, sharex=True)
        for i in range(4):
            # plt.subplot(400+10+i+1)
            # y_major_locator = MultipleLocator(y_kedu[i])
            ax[i].set_ylim(0, y_kedu[i]*1.5)
            ax[i].set_xlim(0, 1)
            ax[i].set_yticks([0,y_kedu[i]*0.5, y_kedu[i]*1])
            # plt.yaxis.set_major_locator(y_major_locator)
            four_like_dis.append(torch.sum(mats[i], dim=1).cpu().numpy())
            ax[i].hist(torch.sum(mats[i], dim=1).cpu().numpy(),**kwargs, label=name[i],)  # num_samples
            ax[i].legend(loc='upper center')
            # plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')
        # print('log_dir',self.training_opt['log_dir'],'type',type(self.training_opt['log_dir']))
        # os.sep() 居然会报错
        f = open(self.training_opt['log_dir'] + '/'+'dis', 'wb')
        pickle.dump(four_like_dis, f)
        f.close()
        fig.savefig(self.training_opt['log_dir'] + '/'+'test_likelihood.png')
        # plt.show()
        # fig.close()

