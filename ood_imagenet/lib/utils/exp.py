from torchvision import transforms


def get_num_labels(dataset):
    return {
        "cifar10": 10,
        "fmnist": 10,
        "mnist": 10,
        "kmnist": 10,
        "cifar100": 100
    }[dataset]


def get_img_size(ind):
    key = ind
    return {
        "cifar10": 32,
        "cifar100": 32,
        "fmnist": 28,
        "mnist": 28,
        "kmnist": 28,
        "svhn": 32,
        "iSUN": 32,
        "Imagenet": 32,
        "Imagenet_resize": 32,
        "LSUN": 32,
        "LSUN_resize": 32
    }[key]


def get_inp_channel(ind):
    key = ind
    return {
        "cifar10": 3,
        "cifar100": 3,
        "fmnist": 1,
        "kmnist": 1,
        "mnist": 1,
        "svhn": 3,
    }[key]

def get_mean(ind):
    key = ind
    return {
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.4914, 0.4822, 0.4465),
        "fmnist": (0.2860,),
        "svhn": (0.4376821046090723, 0.4437697045639686, 0.4728044222297267),
        "mnist": (0.1307, ),
        "kmnist": (0.1307, ),
    }[key]

def get_std(ind):
    key = ind
    return {
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2023, 0.1994, 0.2010),
        "fmnist": (0.3530,),
        "mnist": (0.3081,),
        "kmnist": (0.3081,),
        "svhn": (0.19803012447157134, 0.20101562471828877, 0.19703614172172396),
    }[key]


def get_test_transform(ind):
    return  transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(ind), get_std(ind))])


def get_augmentation_transform(ind):
    augmentation_dict = {
        "cifar10":  transforms.Compose([ transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]), 
        "cifar100":  transforms.Compose([ transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]), 
    }
    return augmentation_dict.get(ind, None)

def get_train_transform(ind):
    key = ind
    return {
        "cifar10": transforms.Compose([ transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key))]),
        "cifar100": transforms.Compose([ transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key))]),
        #"cifar10": transforms.Compose([ transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key))]),
        #"cifar100": transforms.Compose([ transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key))]),
        "svhn": transforms.Compose([ transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "fmnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "kmnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
        "mnist": transforms.Compose([transforms.ToTensor(), transforms.Normalize(get_mean(key), get_std(key)),]),
    }[key]

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name: str = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)