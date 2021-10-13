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
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import yaml
from utils import source_import, get_value

data_root = {'ImageNet': '/home/xxx/imagenet/Img',
             'Places': '/home/xxx/place/places365_standard',
             'iNaturalist18': '/home/xxx/iNaturalist2018'}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--save_feat', type=str, default='')
parser.add_argument('--num_epochs', type=int, default=-1)
parser.add_argument('--keep_classifier', type=int, default=0)  # no use
parser.add_argument('--keep_cls', type=int, default=0)  # no use
parser.add_argument('--stage1_weights_cls', type=bool, default=False)  # no use

# KNN testing parameters 
parser.add_argument('--knn', default=False, action='store_true')
parser.add_argument('--feat_type', type=str, default='cl2n')
parser.add_argument('--dist_type', type=str, default='l2')

# Learnable tau
parser.add_argument('--val_as_train', default=False, action='store_true')
# ---encourage loss---
parser.add_argument('-cr', '--cr', type=float, default=1e-2,
                   help="means courage rate,update rate for courage: "
                        "-1 means  set by  1/(steps per epoch)--not supported currently;"
                        " >0  means manually set the value,")
parser.add_argument('-br', '--bonus_rho', type=float, default=1,
                   help="how much bonus compared with ")
parser.add_argument('-se', '--start_encourage_epoch', type=int, default=1e4, help='when to start encourage training')
parser.add_argument('-as', '--always_stat', type=int, default=0,
                   help='do not update by step,always stat every epoch and encourage')
parser.add_argument('-cg', '--cl_gamma', type=float, default=2, help='step_courage = (1-p)^cl_gamma * y')
parser.add_argument('-fg', '--fl_gamma', type=float, default=1,   # 2020 11 11 修改这里 从默认0 到默认1
                   help='only used to control fl')
parser.add_argument('-cf', '--courage_func', type=str, default='(1-p)^cl_gamma',
                   choices=['(1-p)^cl_gamma', '(cb-p)^cl_gamma', '_p^cl_gamma'])
# parser.add_argument('-ews', '--encourage_warmup_steps', type=int, default=0,
#                    help="")
parser.add_argument('-ewe', '--encourage_warmup_epoch', type=int, default=0,
                   help="")
parser.add_argument('-wg', '--warmup_gamma', type=int, default=1,
                   help="")
parser.add_argument('-bg', '--bonus_gamma', type=float, default=-1,   #6/5 改成-1 之前无论咋样都是log
                   help=" bg>=0  e.g. 2  -logp - -p^2")
# parser.add_argument('-ur', '--updates_round', type=int, default=2048)
parser.add_argument('-cb', '--courage_bound', type=float, default=0.5,
                   help="")
parser.add_argument('--report_class_acc', type=int, default=0,
                   help="")
parser.add_argument('--save_name', type=str, default='', help="")
parser.add_argument('--loss_type', default="CE", type=str, choices=['CE','courage'],help='apply encourage loss or not')
parser.add_argument('--base_loss', default="CE", type=str, choices=['CE','CE_w','LDAM','FL','FL_DRW','FL_w','HFL','LDAM','LDAM_DRW','DRW'],help='base loss in courage loss')

parser.add_argument('--courage_by_weight', type=int, default=0,
                   help="whether use weight for courage")
parser.add_argument('--weight_by_courage', type=int, default=0,
                   help="whether use courage for weight")
parser.add_argument('--norm_courage', type=int, default=0,
                   help="norm courage, especially for weight_by courage,and se160")
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('-cw', '--cw', type=str, default='same',choices=['same','max'],
                   help="") # 2021 2.21 0.31 默认从max改成same
parser.add_argument('-ds','--defer_start', type=int, default=0,
                   help="")
parser.add_argument('-es','--el_start', type=int, default=-1,
                   help="")
parser.add_argument('-beta', '--beta', type=float, default=0.9999,)
parser.add_argument('-pl','--plot_like', type=int, default=0,
                   help="")
parser.add_argument('-load','--load', type=int, default=0,
                   help="")
# parser.add_argument('-pl','--plot_like', type=int, default=0,
#                    help="")

parser.add_argument('--el_part', default=1.0, type=float,help='1.0  all encourage; <1.0  1-el_part common classes'
                                                              ' not encouraged; >1  (el_part-1)2, e.g. 4-1=011 '
                                                              'encourage medium and few ')
parser.add_argument('--bonus_start', default=0.0, type=float,help='default is not piecewise bonus')
parser.add_argument('--uniform_courage', default=0, type=float,help='whether to set uniform courage')
# hfl
parser.add_argument('--mle_bound', default=0.5, type=float,help='used in hfl')
parser.add_argument('--hfl_part', default=1.0, type=float,help='1.0  all encourage; <1.0  1-el_part common classes')
parser.add_argument('-le', '--log_end', type=float, default=1.0,)

args = parser.parse_args()

def update(config, args):
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['log_dir'] = get_value(config['training_opt']['log_dir'], args.log_dir)
    if args.stage1_weights_cls  ==True: # not use
        config['networks']['classifier']['params']['stage1_weights'] = get_value(config['networks']['classifier']['params']['stage1_weights'], args.stage1_weights_cls)
    if args.num_epochs != -1:
        config['training_opt']['num_epochs'] = get_value(config['training_opt']['num_epochs'], args.num_epochs)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)

    # Testing with KNN
    if args.knn and args.test:
        training_opt = config['training_opt']
        classifier_param = {
            'feat_dim': training_opt['feature_dim'],
            'num_classes': training_opt['num_classes'], 
            'feat_type': args.feat_type,
            'dist_type': args.dist_type,
            'log_dir': training_opt['log_dir']}
        classifier = {
            'def_file': './models/KNNClassifier.py',
            'params': classifier_param,
            'optim_params': config['networks']['classifier']['optim_params']}
        config['networks']['classifier'] = classifier
    
    return config

# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.load(f)
config = update(config, args)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config['training_opt']
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.mkdir(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)

def split2phase(split):
    if split == 'train' and args.val_as_train:
        return 'train_val'
    else:
        return split

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                      'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
    else:
        sampler_dic = None

    splits = ['train', 'train_plain', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=split2phase(x), 
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'])
            for x in splits}

    training_model = model(config, data, args, test=False)
    if args.load == 1:
        training_model.load_model(args.model_dir)
    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)

    print('Under testing phase, we load training data simply to calculate \
           training data number for each class.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    if args.knn or True:
        splits.append('train_plain')

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False)
            for x in splits}
    
    training_model = model(config, data, args,test=True)
    # training_model.load_model()
    training_model.load_model(args.model_dir)
    if args.save_feat in ['train_plain', 'val', 'test']:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False
    
    training_model.eval(phase=test_split, openset=test_open, save_feat=saveit)
    
    if output_logits:
        training_model.output_logits(openset=test_open)
        
print('ALL COMPLETED.')
