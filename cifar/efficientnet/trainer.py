from __future__ import print_function
import os
import shutil
from abc import ABCMeta, abstractmethod
import codecs
import mlconfig
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from .metrics import Accuracy, Average
from .el import EncourageLoss
from .models import EfficientNetL, EfficientNetLS
from torch.cuda.amp import GradScaler

import numpy as np
import matplotlib.pyplot as plt

class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

# class ARGS():
#     def __init__(self, bonus_gamma=0.0, bonus_rho=1.0, defer_start=0,bonus_start=0.0,):
#         self.bonus_gamma = bonus_gamma
#         self.bonus_rho = bonus_rho
#         self.defer_start = defer_start
#         self.bonus_start = bonus_start

@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, train_loader: data.DataLoader,
                 valid_loader: data.DataLoader, scheduler: optim.lr_scheduler._LRScheduler, device: torch.device,
                 num_epochs: int, output_dir: str, args=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.epoch = 1
        self.best_acc = 0
        self.args = args
        # self.bonus_gamma=bonus_gamma
        # self.bonus_rho=bonus_rho
        # args = ARGS(bonus_gamma=bonus_gamma, bonus_rho=bonus_rho)
        self.el = None
        self.multi_margin_loss=None
        # if args.bonus_gamma != 0:
        #     print('bonus_gamma', args.bonus_gamma)
        #     self.el = EncourageLoss(opt=args).cuda(device)
        #2021 3.12 0.08 想改成全走你encourageloss,  区别顶多是
        if args.MultiMarginLossM > 0:
            self.multi_margin_loss = torch.nn.MultiMarginLoss(margin=args.MultiMarginLossM)
        else:
            self.el = EncourageLoss(opt=args).cuda(device)



    def fit(self):
        epochs = trange(self.epoch, self.num_epochs + 1, desc='Epoch', ncols=0)
        log_dir = 'experiments/' + self.args.log + '.txt'
        f_log = codecs.open(filename=log_dir, mode='w+')
        print(self.model)
        print(self.model, file=f_log)
        print('| num. model params: {} (num. trained: {})'.format(
            sum(p.numel() for p in self.model.parameters()),
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        ))
        print('| num. model params: {} (num. trained: {})'.format(
            sum(p.numel() for p in self.model.parameters()),
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        ), file=f_log)
        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler() if self.args.amp == 1 else None
        for self.epoch in epochs:
            if self.el is not None:
                self.el.set_epoch(self.epoch)
            self.scheduler.step()

            train_loss, train_reward, train_acc = self.train(scaler=scaler)
            valid_loss, valid_reward, valid_acc = self.evaluate()

            self.save_checkpoint(os.path.join(self.output_dir, 'checkpoint.pth'))
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc.value
                self.save_checkpoint(os.path.join(self.output_dir, 'best.pth'))

            epochs.set_postfix_str(f'epoch: {self.epoch:3d}, best valid acc: {self.best_acc:.2f},'
                  f'train loss: {train_loss},train_reward:{train_reward}, train acc: {train_acc}, '
                  f'valid loss: {valid_loss},train_reward:{valid_reward}, valid acc: {valid_acc}, ')
            print(f'epoch: {self.epoch:3d}, best valid acc: {self.best_acc:.2f},'
                  f'train loss: {train_loss},train_reward:{train_reward}, train acc: {train_acc}, '
                  f'valid loss: {valid_loss},train_reward:{valid_reward}, valid acc: {valid_acc}, ',
                  file=f_log)

    def train(self,scaler=None):
        self.model.train()

        train_loss = Average()
        train_reward = Average()
        train_acc = Accuracy()

        train_loader = tqdm(self.train_loader, ncols=0, desc='Train')
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            if isinstance(self.model, EfficientNetL) or isinstance(self.model, EfficientNetLS):
                output = self.model(x,y)
            else:
                output = self.model(x)
                if self.args.margin > 1:
                    output = self.model.ls(output, y)
            if self.el is None:
                if self.multi_margin_loss is not None:
                    loss = self.multi_margin_loss(output, y)
                else:
                    loss = F.cross_entropy(output, y)
                train_loss.update(loss.item(), number=x.size(0))
                reward = torch.Tensor([0])
                train_reward.update(reward.item(), number=x.size(0))
            else:
                all_loss, mle_loss = self.el(output, y)
                train_loss.update(mle_loss.item(), number=x.size(0))
                loss = all_loss
                reward = all_loss - mle_loss
                train_reward.update(reward.item(), number=x.size(0))

            self.optimizer.zero_grad()
            if scaler is not None:
                # Scales the loss, and calls backward() on the scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(self.optimizer)
                # Updates the scale for next iteration.
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_acc.update(output, y)

            train_loader.set_postfix_str(f'train loss: {train_loss}, train reward: {train_reward}, train acc: {train_acc}.')

        return train_loss, train_reward, train_acc

    def evaluate(self):
        self.model.eval()

        valid_loss = Average()
        valid_reward = Average()
        valid_acc = Accuracy()

        with torch.no_grad():
            valid_loader = tqdm(self.valid_loader, desc='Validate', ncols=0)
            for x, y in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                if self.el is None:
                    loss = F.cross_entropy(output, y)
                    valid_loss.update(loss.item(), number=x.size(0))
                    reward = torch.Tensor([0])
                    valid_reward.update(reward.item(), number=x.size(0))
                else:
                    all_loss, mle_loss = self.el(output, y)
                    valid_loss.update(mle_loss.item(), number=x.size(0))
                    reward = mle_loss - all_loss
                    valid_reward.update(reward.item(), number=x.size(0))
                valid_acc.update(output, y)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid_reward:{valid_reward}, valid acc: {valid_acc}.')

        return valid_loss, valid_reward, valid_acc

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']

    def resume_and_only_tune_classifier(self,f):
        checkpoint = torch.load(f, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.args.model.startswith('resnet50'):
            print('Re-init the output layer. model: ',self.args.model)
            self.model.fc.reset_parameters()
        elif self.args.model.startswith('efficientnet'):
            print('Re-init the output layer. model: ',self.args.model)
            self.model.classifier[1].reset_parameters()
        print('And freeze other parameters.')
        for param_name, param in self.model.named_parameters():
            # Freeze all parameters except self attention parameters
            if 'fc' not in param_name and 'classifier' not in param_name:
                param.requires_grad = False

    def plots_cifar(self,dir_name='cifar10_plot',num_classes=10,which_data='valid',which_model='resnet50'):  # consider convenience, only for cifar10
        c10_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        cifar_idx = range(num_classes)
        total_logits = torch.empty((0, num_classes)).cuda()
        if which_model.startswith('efficientnet'):
            total_reps = torch.empty((0, 2048)).cuda()
        else:
            total_reps = torch.empty((0, 1280)).cuda()
        total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.model.eval()
        valid_acc = Accuracy()
        if which_data=='valid':
            valid_loader = tqdm(self.valid_loader, desc='Validate', ncols=0)
        else:
            valid_loader = tqdm(self.train_loader, desc='Validate', ncols=0)
        dir_name=dir_name+'_'+which_data
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                if which_model.startswith('efficientnet'):
                    output_reps = eb0_forward(self.model,x)
                    total_reps = torch.cat((total_reps, output_reps))
                else:
                    output_reps = r50_forward_without_fc(self.model,x)
                    total_reps = torch.cat((total_reps, output_reps))
                output = self.model(x)  # bsz * num_classes
                total_logits = torch.cat((total_logits, output))
                total_labels = torch.cat((total_labels, y))
                # loss = F.cross_entropy(output, y)
                valid_acc.update(output, y)
        org_probs = F.softmax(total_logits, dim=1)
        _, total_preds = org_probs.max(dim=1)
        pred_hot = F.one_hot(total_preds.view(-1), num_classes=num_classes)  # bsz, num_classes
        targets_hot = F.one_hot(total_labels.view(-1), num_classes=num_classes)  # bsz, num_classes

        # 0. plot tsne         # self.plot_tsne()
        # self.sh_tsne(total_logits.cpu().numpy(),total_labels.cpu().numpy(), figure_name=dir_name+'classifier_tsne',c10_names=c10_names)
        # self.sh_tsne(total_reps.cpu().numpy(), total_labels.cpu().numpy(), figure_name=dir_name + 'reps_tsne',c10_names=c10_names)
        # 1 print acc, ,class_name, class level acc,
        print(valid_acc)
        class_acc = self.class_acc(pred_hot,targets_hot)
        for ci in cifar_idx:
            print(c10_names[ci],class_acc[ci].cpu()*100)

        # 2 give likelihood dis
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        self.plot_likelihood(org_probs,targets_hot,dir_name)

        # 3. calculate ECE
        # uncalibrated_score = self.cal_ece(org_probs.cpu().numpy(),total_labels.cpu().numpy(),dir_name)
        print('uncalibrated_score:', ece_score(org_probs.cpu().numpy(),total_labels.cpu().numpy(),10))

        # 4. calculate margin distribution
        self.plot_margin(total_logits, targets_hot, dir_name)

        # 5. plot class-level energy distribution, on 10/100 sample figures and print accuracy for each class
        self.plot_energy(total_logits, total_labels, dir_name,c10_names)
        return

    def class_acc(self,pred_hot,targets_hot):
        class_correct = torch.sum(pred_hot.float() * pred_hot.eq(targets_hot).float(), dim=0)
        class_total = torch.sum(targets_hot, dim=0).float()
        class_acc = class_correct / class_total
        return class_acc

    def plot_likelihood(self,org_probs,targets_hot,dir_name):
        like_mat = org_probs * targets_hot.float()  # num_samples, num_classes
        kwargs = dict(alpha=0.5, bins=20)
        # from matplotlib.backends.backend_pdf import PdfPages
        # pdf = PdfPages('iNaturalist2018_90' + right_names[j] + '_test_likelihood.pdf')
        fig, ax = plt.subplots(1, 1,)
        ax.set_yticks([0, 10000 * 0.10, 10000 * 1])
        ax.set_xlabel('likelihood')
        ax.hist(torch.sum(like_mat, dim=1).cpu().numpy(),**kwargs, label='likelihood distribution',)  # num_samples
        ax.legend(loc='upper center')
        fig.savefig(dir_name + '/' + 'likelihood.png')
        # f = open(dir_name + '/'+'dis', 'wb')
        # import pickle
        # pickle.dump(like_mat, f)
        # pdf.savefig()
        plt.close()
        # pdf.close()

    def cal_ece(self,dir_name,confidences,ground_truth):
        from netcal.metrics import ECE
        from netcal.presentation import ReliabilityDiagram
        n_bins = 10
        ece = ECE(n_bins)
        # uncalibrated_score = ece.measure(confidences)
        uncalibrated_score = 0
        # calibrated_score = ece.measure(calibrated)
        diagram = ReliabilityDiagram(n_bins)
        diagram.plot(confidences, ground_truth,filename=dir_name + '/' + 'ece.png')  # visualize miscalibration of uncalibrated
        # diagram.plot(calibrated, ground_truth)  # visualize miscalibration of calibrated
        return uncalibrated_score

    def plot_margin(self,total_logits,targets_hot,dir_name):
        logits_mat = total_logits * targets_hot.float()  # num_samples, num_classes
        max_negative_logits, indices=torch.max(total_logits * (torch.ones_like(targets_hot)-targets_hot).float(),dim=1) # num_samples
        target_logits=torch.sum(logits_mat, dim=1)
        margin_mat = (target_logits-max_negative_logits).cpu().numpy()  # num_samples each with a margin
        kwargs = dict(alpha=0.5, bins=20)
        fig, ax = plt.subplots(1, 1,)
        ax.set_xlabel('margin')
        ax.hist(margin_mat,**kwargs, label='margin distribution',)  # num_samples
        ax.legend(loc='upper left',fontdict={'fontsize': 25})
        fig.savefig(dir_name + '/' + 'margin.png')
        plt.close()

    def plot_energy(self,total_logits, total_labels, dir_name, c10_names):  # class level energy
        energy = -total_logits
        cifar_idx= range(len(c10_names))
        for i in cifar_idx:
            energy_i = torch.mean(energy[total_labels == i, :], dim=0).cpu()  # num_classes
            print('i:',i,'| name:', c10_names[i],)
            # print(str(energy_i.tolist()))
            energy_i_sorted, energy_i_indices = torch.sort(energy_i)
            # sorted_names = [c10_names[energy_i_indices[j]] for j in cifar_idx]
            # print('energy_i_indices', energy_i_indices)
            # print('sorted_names',sorted_names)
            print('energy_i_sorted',energy_i_sorted)
            print('name-energy', " ".join([c10_names[energy_i_indices[j]] +':' + "%.2f" % (energy_i[energy_i_indices[j]].item()) for j in cifar_idx]))


        # cnt = 0
        # plt.figure(figsize=(8, 10))
        # for i in range(len(epsilons)):
        #     for j in range(len(examples[i])):
        #         cnt += 1
        #         plt.subplot(len(epsilons), len(examples[0]), cnt)
        #         plt.xticks([], [])
        #         plt.yticks([], [])
        #         if j == 0:
        #             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        #         orig, adv, ex = examples[i][j]
        #         plt.title("{} -> {}".format(orig, adv))
        #         plt.imshow(ex, cmap="gray")
        # plt.tight_layout()
        # plt.show()

    def sh_tsne(self,x, y, figure_name,c10_names):
        '''
        x is a datamat, type is ndarray, shape(x)[0] reps the number of data, shape(x)[1] reps the ori dim of a data
        y is labels, type is array, shape(y)[0] reps the number of labels of data.
        '''
        from sklearn import manifold
        colors=[
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        x_tsne = tsne.fit_transform(x)
        print("Org data dimension is {}. Embedded data dimension is {}".format(x.shape[-1], x_tsne.shape[-1]))
        x_min, x_max = x_tsne.min(0), x_tsne.max(0)
        x_norm = (x_tsne - x_min) / (x_max - x_min)  # normalization
        plt.figure(figsize=(8, 8))
        for i in range(x_norm.shape[0]):
            plt.text(x_norm[i, 0], x_norm[i, 1], c10_names[y[i]], color=colors[y[i]],
                     fontdict={'weight': 'bold', 'size': 9})
        plt.title('t-sne')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(figure_name+'.pdf')

        # plt.show()
    def deli_tsne(self,outputs, labels, figure_name,c10_names):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import os
        import numpy as np
        # import torch

        color = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        ave = lambda x: sum(x) / len(x)

        def draw( label_name=c10_names, classes=10, path=figure_name):
            E = outputs
            N = E.shape[0]  # number of samples
            E = TSNE(n_components=2).fit_transform(E)
            # torch.save(E,'E.pt')
            # E = torch.load('E.pt')
            # N = E.shape[0]
            # label = indict['label']
            label = labels
            # pca.fit(E)
            # E = pca.transform(E)
            x = [[] for i in range(classes)]
            y = [[] for i in range(classes)]
            for i in range(N):
                x[label[i]].append(float(E[i, 0])) # 给每类
                y[label[i]].append(float(E[i, 1]))
            for i in range(classes):
                plt.scatter(x[i], y[i], color=color[i], s=1, alpha=0.5)  # 画出来散点位置
            plts = []
            for i in range(classes):
                plts.append(plt.scatter(ave(x[i]), ave(y[i]), color=color[i], s=10, alpha=1.0))  # 画出来类别原型位置
            plt.legend(plts, label_name)
            plt.savefig(path)
            plt.cla()
        draw()

# import numpy as np


def ece_score(py, y_test, n_bins=10):
    # py = np.array(py)
    # y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


def r50_forward_without_fc(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    return x

def eb0_forward(self, x):
    x = self.features(x)
    x = x.mean([2, 3])
    return x