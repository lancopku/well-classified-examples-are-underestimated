import argparse

import mlconfig
import torch
from torch import distributed, nn

from efficientnet.utils import distributed_is_initialized
import torchvision.models as models
from efficientnet.models import LSoftmaxLinear

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/mnist.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--data-parallel', action='store_true')

    # distributed
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23455')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--log', '-log', type=str, default='efficientnetB0', help='')
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help='')
    parser.add_argument('--defer_start','-ds', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0125)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bonus_gamma', '-bg',type=float, default=0)
    parser.add_argument('--bonus_start', '-bs',default=0.0, type=float, help='default is not piecewise bonus')
    parser.add_argument('--bonus_rho', '-br',type=float, default=1.0)
    parser.add_argument('--log_end', '-le',type=float, default=1.0)
    parser.add_argument('--margin', type=int, default=1,help='for efficientnetl_b0')
    parser.add_argument('--model', type=str, default='efficientnet_b0')
    parser.add_argument('--base_loss', '-bl',type=str, default='ce',
                        help='ce|mse|mae|mse_sigmoid|mae_sigmoid')
    # 2021 3.29 i think label smoothing could also be applied to the encouraging loss
    parser.add_argument('--el_lb', type=float, default=0)
    parser.add_argument('--el_mask_pad', type=int, default=0)  # should be set to 1
    # 2021 3.31 add schedule
    parser.add_argument('--el_schedule', type=str, choices=[None, 'linear', 'exp', 'cos'], default=None)
    parser.add_argument('--el_schedule_max_epoch', type=int,
                        default=None)
    parser.add_argument('--el_linear_schedule_start', type=int, default=0)
    parser.add_argument('--el_linear_schedule_end', type=int, default=None)
    parser.add_argument('--el_exp_schedule_ratio', type=int, default=5)

    parser.add_argument('--amp', type=int, default=0,help='')
    parser.add_argument('--sub_data_rho', type=float, default=1,help='')
    # 4.20
    parser.add_argument('-les','--el_schedule_le', type=int, default=0)
    parser.add_argument('-resume_and_only_tune_classifier', '--resume_and_only_tune_classifier',  type=str, default=None)
    parser.add_argument('-MultiMarginLossM', '--MultiMarginLossM', type=float, default=0.0)
    parser.add_argument('--plots_cifar', type=str, default=None)
    parser.add_argument('--plots_cifar_data', type=str, default='valid')
    parser.add_argument('--bonus_abrupt', type=float, default=-1)
    parser.add_argument('--label-smoothing', type=float, default=0,
                        help='')
    parser.add_argument('--plot_gnorm', type=float, default=0,
                        help='')


    return parser.parse_args()


def init_process(backend, init_method, world_size, rank):
    distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

def main():
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    config = mlconfig.load(args.config)


    # reset config # 2021 1.30 16:57
    config.dataset.batch_size = args.batch_size  # 32
    config.optimizer.lr = args.lr  #  0.0125
    config.trainer.output_dir='experiments/' + args.log # experiments / cifar10

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed:
        init_process(args.backend, args.init_method, args.world_size, args.rank)

    # device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    device = torch.device(args.gpu)

    if args.model.startswith('efficientnet'):
        if args.model.startswith('efficientnetl'):
            config.model.name = args.model
            config.model.margin = args.margin
            config.model.gpuid = args.gpu
        model = config.model()
    else:
        num_classes = config.model.num_classes
        model = models.__dict__[args.model](num_classes=num_classes)
        if args.margin>1:
            model.ls=LSoftmaxLinear(model.fc.weight.size()[1],model.fc.weight.size()[0],margin=args.margin,device=device)
            model.ls.reset_parameters()
            model.fc=nn.Sequential()
    print(args)
    print(config)
    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        if args.data_parallel:
            model = nn.DataParallel(model)
        model.to(device)

    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    train_loader = config.dataset(train=True)
    valid_loader = config.dataset(train=False)

    trainer = config.trainer(model, optimizer, train_loader, valid_loader, scheduler, device, args=args)

    if args.resume is not None:
        trainer.resume(args.resume)
    if args.resume_and_only_tune_classifier is not None:
        trainer.resume_and_only_tune_classifier(args.resume_and_only_tune_classifier)
    if args.plots_cifar is not None:
        trainer.plots_cifar(dir_name=args.plots_cifar,which_data=args.plots_cifar_data,which_model=args.model)
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
