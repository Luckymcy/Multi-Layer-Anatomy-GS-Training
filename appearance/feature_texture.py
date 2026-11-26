"""Frozen VGG16 shallow feature SW with receptive-field-safe tissue masks."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16


class MaskedFeatureSW(nn.Module):
    def __init__(self, weights):
        super().__init__()
        network=vgg16(pretrained=False)
        network.load_state_dict(torch.load(weights,map_location='cpu'))
        self.features=network.features[:9].eval().requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        generator=torch.Generator().manual_seed(731)
        for channels in (64,128):
            self.register_buffer(f'directions_{channels}',F.normalize(torch.randn(32,channels,generator=generator),dim=1))

    def tokens(self, srgb, mask, maximum=2048):
        value=(srgb[None]-self.mean)/self.std
        valid=mask[None,None].float()
        result={}
        for index,layer in enumerate(self.features):
            value=layer(value)
            if isinstance(layer,nn.Conv2d):
                valid=1-F.max_pool2d(F.pad(1-valid,(1,1,1,1),value=1),3,stride=1)
            elif isinstance(layer,nn.MaxPool2d):
                valid=1-F.max_pool2d(1-valid,2,stride=2)
            if index in (3,8):
                tokens=value[0].permute(1,2,0)[valid[0,0]>.999]
                if len(tokens)>=32:
                    ids=torch.randperm(len(tokens),device=tokens.device)[:maximum]
                    result[index]=tokens[ids]
        return result

    def loss(self, fake, real):
        losses=[]
        for level in fake.keys() & real.keys():
            a,b=fake[level],real[level];count=min(512,len(a),len(b))
            a=a[torch.randperm(len(a),device=a.device)[:count]]
            b=b[torch.randperm(len(b),device=b.device)[:count]]
            direction=getattr(self,f'directions_{a.shape[1]}')
            losses.append(F.mse_loss((a@direction.T).sort(0).values,(b@direction.T).sort(0).values))
        if not losses: raise ValueError('No common valid shallow feature level')
        return torch.stack(losses).mean()
