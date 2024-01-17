import types
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet152
from KD_Lib.models.resnet_old import ResNet18, ResNet50, ResNet152

def _forward_impl(self, x: torch.Tensor, return_feats: bool=False, norm_feats: bool=False) -> torch.Tensor:
    feats = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    feats.append(x)
    x = self.layer1(x)
    feats.append(x)
    x = self.layer2(x)
    feats.append(x)
    x = self.layer3(x)
    feats.append(x)
    x = self.layer4(x)
    feats.append(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    if norm_feats:
        xl2 = torch.norm(x, p=2, dim=1, keepdim=True)
        xl2 = torch.clip(xl2,0,8)
        x1 = F.normalize(x, p=2, dim=1)
        x = torch.mul(x1,xl2)
    feats.append(x)
    out = self.fc(x)
    if return_feats:
        return out, feats, self.fc.weight, self.fc.bias
    return out

def _forward_impl_custom(self, x: torch.Tensor, return_feats: bool=False, norm_feats: bool=False) -> torch.Tensor:
    feats = []
    out = F.relu(self.bn1(self.conv1(x)))
    feats.append(out)
    out = self.layer1(out)
    feats.append(out)
    out = self.layer2(out)
    feats.append(out)
    out = self.layer3(out)
    feats.append(out)
    out = self.layer4(out)
    feats.append(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    if norm_feats:
        xl2 = torch.norm(out, p=2, dim=1, keepdim=True)
        xl2 = torch.clip(xl2,0,8)
        x1 = F.normalize(out, p=2, dim=1)
        out = torch.mul(x1,xl2)
    feats.append(out)
    out = self.linear(out)
    if return_feats:
        return out, feats, self.linear.weight, self.linear.bias
    return out

def _forward_impl_custom_new(self, x: torch.Tensor, return_feats: bool=False, norm_feats: bool=False) -> torch.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)  # 32x32
    f0 = x

    x, _ = self.layer1(x)  # 32x32
    f1 = x
    x, _ = self.layer2(x)  # 16x16
    f2 = x
    x, _ = self.layer3(x)  # 8x8
    f3 = x

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    f4 = x
    x = self.fc(x)

    if norm_feats:
        # xl2 = torch.norm(f4, p=2, dim=1, keepdim=True)
        # xl2 = torch.clip(xl2,0,2)
        # x1 = F.normalize(f4, p=2, dim=1) # x1 
        # xn = torch.mul(x1,xl2)
        # xn = self.fc(xn)
        xl2 = torch.norm(f4, p=2, dim=1, keepdim=True)
        xn = torch.div(f4, xl2) * 2
        xn = self.fc(xn)
        x = (x, xn)
    
    if return_feats:
        return x, [f0, f1, f2, f3, f4], self.fc.weight, self.fc.bias
    else:
        return x

def monkey_patch(model, custom=False):
    if custom:
        model.forward = types.MethodType(_forward_impl_custom_new, model)
    else:
        model.forward = types.MethodType(_forward_impl, model)
    return model

def get_ResNet(model, cfg):
    if model == 'resnet18':
        model_class = resnet18 if not cfg.CUSTOM_MODEL else ResNet18
    elif model == 'resnet50':
        model_class = resnet50 if not cfg.CUSTOM_MODEL else ResNet50
    elif model == 'resnet152':
        model_class = resnet152 if not cfg.CUSTOM_MODEL else ResNet152
    else:
        raise NotImplementedError
    if cfg.CUSTOM_MODEL:
        model = model_class(num_classes=cfg.CLASSES, weights=None)
    else:
        model = model_class(num_classes=cfg.CLASSES, weights=None)

    model = monkey_patch(model, custom=cfg.CUSTOM_MODEL)
    if cfg.PARALLEL:
        model = torch.nn.DataParallel(model)
    return model