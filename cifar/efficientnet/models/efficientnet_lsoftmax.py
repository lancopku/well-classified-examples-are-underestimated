import math

import mlconfig
import torch
from torch import nn

from .utils import load_state_dict_from_url
from .efficientnet import _round_filters,ConvBNReLU,Swish,SqueezeExcitation,MBConvBlock,model_urls,params,_round_repeats
from .lsoftmax import LSoftmaxLinear
from  torch.nn import functional as F

@mlconfig.register
class EfficientNetL(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000, gpuid=0, margin=1):
        super(EfficientNetL, self).__init__()
        self.device=torch.device(gpuid)
        # yapf: disable
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]
        # yapf: enable

        out_channels = _round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]

        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels

        last_channels = _round_filters(1280, width_mult)
        features += [ConvBNReLU(in_channels, last_channels, 1)]

        self.features = nn.Sequential(*features)
        self.dropout_rate=dropout_rate
        self.classifier = LSoftmaxLinear(input_features=last_channels, output_features=num_classes, margin=margin, device=self.device)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LSoftmaxLinear):
                m.reset_parameters()

    def forward(self, x,target=None):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(F.dropout(x, self.dropout_rate), target=target)
        return x

def _efficientnetl(arch, pretrained, progress, gpuid=0, margin=1,**kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNetL(width_mult, depth_mult, dropout_rate,gpuid=gpuid, margin=margin, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
    return model


@mlconfig.register
def efficientnetl_b0(pretrained=False, progress=True, gpuid=0, margin=1, **kwargs):
    return _efficientnetl('efficientnet_b0', pretrained, progress,gpuid=gpuid, margin=margin, **kwargs)