from .lenet import *
from .resnet import *
from ..utils.exp import get_num_labels

def get_model(args):
    model_type = args.model 
    num_classes = get_num_labels(args.dataset)
    if model_type == 'lenet':
        return LeNet(num_classes)
    elif model_type == 'resnet18':
        return ResNet18(num_c=num_classes)
    elif model_type == 'resnet34':
        return ResNet34(num_c=num_classes)
    else:
        raise NotImplementedError