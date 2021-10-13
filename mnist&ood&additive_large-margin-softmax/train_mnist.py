from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from models import MNISTNet, MNISTFIG2Net, NetLayer
from el import EncourageLoss
import time


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    train_ce=0
    train_reward=0
    cnt=0
    for batch_idx, (data, target) in enumerate(train_loader):
        cnt += 1
        data, target = data.to(device), target.to(device)
        if args.vis:
            output, _ = model(x=data, target=target)
        else:
            output = model(x=data, target=target)
        # if args.bonus_gamma != 0:
        loss, org_loss = criterion(input=output, target=target)
        reward = loss-org_loss
        # else:
        #     loss = criterion(input=output, target=target)
        #     reward = torch.zeros_like(loss)
        #     org_loss = loss
        optimizer.zero_grad()
        # clip_grad_norm_(model.parameters(), max_norm=10)
        loss.backward()
        optimizer.step()
        train_ce += org_loss.item()
        train_reward += reward.item()
        if args.print_every_epoch:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] \tBase: {:.6f} \tReward: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), org_loss.item(),reward.item()))
    return train_ce/cnt, train_reward/cnt


def test(args, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_reward=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.vis:
                output, _ = model(x=data)
            else:
                output = model(x=data)
            # if args.bonus_gamma != 0:
            loss, org_loss = criterion(input=output, target=target)
            reward = loss - org_loss
            # else:
            #     loss, org_loss = criterion(input=output, target=target)
            #     reward = torch.zeros_like(loss)
            test_loss += org_loss.item()
            test_reward += reward.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_reward /= len(test_loader.dataset)
    test_acc = 100. * float(correct) / len(test_loader.dataset)
    if args.print_every_epoch:
        print('\nTest set: Average org_loss: {:.6f},reward: {:.6f} Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, test_reward,correct, len(test_loader.dataset),
            test_acc))
    return test_acc,test_loss, test_reward


def plot_2d_features(args, model, device, test_loader):
    net_logits = np.zeros((10000, 2), dtype=np.float32)
    net_labels = np.zeros((10000,), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for b_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            _, output2d = model(x=data)
            output2d = output2d.cpu().data.numpy()
            target = target.cpu().data.numpy()
            net_logits[b_idx * args.test_batch_size: (b_idx + 1) * args.test_batch_size, :] = output2d
            net_labels[b_idx * args.test_batch_size: (b_idx + 1) * args.test_batch_size] = target
        for label in range(10):
            idx = net_labels == label
            plt.scatter(net_logits[idx, 0], net_logits[idx, 1])
        plt.legend(np.arange(10, dtype=np.int32))
        plt.show()


def adjust_learning_rate(args, optimizer, epoch):
    # Decreasing the learning rate to the factor of 0.1 at epochs 51 and 65
    # with a batch size of 256 this would comply with changing the lr at iterations 12k and 15k
    if 50 < epoch < 65:
        lr = args.lr * 0.1
    elif epoch >= 65:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='PyTorch L-Softmax MNIST Example')
    parser.add_argument('--margin', type=int, default=4, metavar='M',
                        help='the margin for the l-softmax formula (m=1, 2, 3, 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--vis', default=False, metavar='V',
                        help='enables visualizing 2d features (default: False).')
    parser.add_argument('--label_smoothing', type=float, default=0,
                        help='')
    # defer_start  bonus_start bonus_gamma bonus_rho
    parser.add_argument('--defer_start', type=int, default=0,
                        help='')
    parser.add_argument('--bonus_start', type=float, default=0,
                        help='')
    parser.add_argument('--bonus_gamma', type=int, default=0,
                        help='')
    parser.add_argument('--bonus_rho', type=float, default=1,
                        help='')
    parser.add_argument('--log_end', type=float, default=1,
                        help='')
    parser.add_argument('--runs', type=int, default=1,
                        help='')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='-1 means cpu')
    parser.add_argument('--print_every_epoch', type=int, default=0,
                        help='whether to print at every epoch')
    parser.add_argument('--print_every_run', type=int, default=0,
                        help='whether to print at every epoch')
    parser.add_argument('--base_loss', type=str, default='ce',
                        help='ce|mse|mae|mse_sigmoid|mae_sigmoid')
    parser.add_argument('--n_layer', type=int, default=0,
                        help='0 means we not study layers in mlp')
    parser.add_argument('--save', type=int, default=0,
                        help='1 means save checkpoint')
    # parser.add_argument('--n_conv', type=int, default=3,
    #                     help='<3 means we want to build small mnistnet')
    args = parser.parse_args()
    print(args)
    runs_ar = np.zeros((args.runs, 7))

    for turn in range(args.runs):
        begin = time.perf_counter()
        use_cuda = args.gpu >= 0
        if args.seed < args.runs:
            seed = turn
        else:
            seed = args.seed
        torch.manual_seed(seed)

        device = torch.device(args.gpu) if use_cuda else  torch.device('cpu')

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        # if args.bonus_gamma != 0:
        criterion = EncourageLoss(opt=args).to(device)
        # else:
            # criterion = nn.CrossEntropyLoss().to(device)
        if args.n_layer>0:
            model = NetLayer(hidden=256, dropout=0.0, layer=args.n_layer).to(device)
        else:
            if args.vis:
                model = MNISTFIG2Net(margin=args.margin, device=device).to(device)
            else:
                model = MNISTNet(margin=args.margin, device=device).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        best_test_acc = 0
        best_epoch = 0
        best_test_train_ce = 0
        best_test_train_reward = 0
        best_test_test_ce = 0
        best_test_test_reward = 0
        best_test_test_acc = 0
        each_run_log_np = np.zeros(7)
        best_model=None
        for epoch in range(1, args.epochs + 1):
            # if args.bonus_gamma !=0:
            criterion.set_epoch(epoch)
            adjust_learning_rate(args, optimizer, epoch)
            train_ce,  train_reward = train(args, model, criterion, device, train_loader, optimizer, epoch)
            test_acc, test_ce, test_reward = test(args, model, criterion, device, test_loader)
            if test_acc > best_test_acc:
                best_test_test_acc = test_acc
                if args.print_every_epoch:
                    print('new best test: ', test_acc)
                best_test_acc = test_acc
                best_epoch = epoch
                best_test_train_ce = train_ce
                best_test_train_reward = train_reward
                best_test_test_ce = test_ce
                best_test_test_reward = test_reward
                best_model=model

        each_run_log_np[0]=best_epoch
        each_run_log_np[1]=best_test_train_ce
        each_run_log_np[2]=best_test_train_reward
        each_run_log_np[3]=best_test_test_acc
        each_run_log_np[4]=best_test_test_ce
        each_run_log_np[5]=best_test_test_reward
        end = time.perf_counter()
        each_run_log_np[6] = end-begin
        run_log = 'Epoch: {:03d}, Train base: {:.4f} reward: {:.4f}, Test: Acc: {:.2f}  base: {:.6f} , reward: {:.6f} time: {:.1f} '. \
            format(best_epoch, best_test_train_ce, best_test_train_reward, best_test_test_acc, best_test_test_ce, best_test_test_reward,each_run_log_np[6])
        if args.print_every_run:
            print(run_log)


        if args.vis:
            plot_2d_features(args, model, device, test_loader)

        runs_ar[turn] = each_run_log_np
        if args.save:
            file_name= 'bg' + str(args.bonus_gamma) +'_le'+str(args.log_end) + '_t'  + str(turn) +'_m'+str(args.margin)+'_ls%.1f'% (args.label_smoothing)  +"_acc%.2f" %best_test_test_acc +'_best.pth'
            torch.save(best_model.state_dict(),file_name)

    final_log = 'Epoch: {:3.1f}, Train base: {:.6f} reward: {:.6f}, Test: Acc: {:.2f}  base: {:.6f} , reward: {:.6f} time: {:.1f} std: {:.2f}'. \
        format(*(list(np.mean(runs_ar, axis=0))), np.std(runs_ar[:,3]))
    print(final_log)
if __name__ == '__main__':
    main()
