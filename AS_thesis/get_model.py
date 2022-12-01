from torchvision.models.video import r2plus1d_18
import torch.nn as nn
import torch.nn.functional as F
from tvn.model import TVN, no_verbose
from tvn.config import CFG1
# from pytorchvideo.models import x3d, resnet, slowfast


class Two_head(nn.Module):
    def __init__(self, model, num_classes1, num_classes2):
        super(Two_head, self).__init__()
        self.model = model
        self.nc1 = num_classes1
        self.nc2 = num_classes2

    def forward(self, x):
        x = self.model(x)
        out1 = x[:, :self.nc1]
        out2 = x[:, self.nc1:self.nc1 + self.nc2]
        return out1, out2


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='r2plus1d_18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.model = r2plus1d_18(pretrained=False, num_classes=2)
        dim_in = self.model.fc.in_features
        self.linear = (head == 'linear')
        if head == 'linear':
            self.model.fc = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'mlp':
            self.model.fc = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.model(x)
        feat = F.normalize(feat, dim=1)
        return feat


class Linear(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='r2plus1d_18', nc=4):
        super(Linear, self).__init__()
        self.model = r2plus1d_18(pretrained=False, num_classes=nc)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        feat = self.model(x)
        return feat


def get_model(config):
    if config['loss_type'] == 'laplace_cdf':
        nc = 2
    else:
        nc = config['num_classes']

    if config['cotrastive_method'] == "CE":
        if config['model'] == "ava_r2plus1d_18":
            model = r2plus1d_18(pretrained=config['pretrained'], num_classes=1)
        if config['model'] == "r2plus1d_18":
            # instantiate the pretrained model
            model = r2plus1d_18(pretrained=config['pretrained'], num_classes=nc)
            # model = Two_head(model, nc, 2)
        if config['model'] == "tvn":
            model = TVN(CFG1, nc)
        # elif config['model'] == "resnet50":
        #     model = resnet.create_resnet(
        #         input_channel=3,
        #         model_depth=50,
        #         model_num_class=config['num_classes'])
    elif config['cotrastive_method'] == "SupCon" or config['cotrastive_method'] == "SimCLR":
        model = SupConResNet(head='mlp', feat_dim=config['feature_dim'])
    elif config['cotrastive_method'] == "Linear":
        model = Linear(nc=4)
    else:
        raise NotImplementedError('contrastive_method not supported: {}')

    # Calculate number of parameters.
    num_parameters = sum([x.nelement() for x in model.parameters()])
    print(f"The number of parameters in {config['model']}: {num_parameters / 1000:9.2f}k")

    return model
