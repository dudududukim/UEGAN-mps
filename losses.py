import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn
import torchvision.models as models
import os
from math import pi


# ==============================
# (2) fidelity loss (in paper) (raw to exp)
#     -> 일반적인 pixel by pixel 비교랑 달라서 perceptualLoss로 명명한듯
# Lfid = j=1{Exl ∼Pl ∥φj (xl)−φj (G(xl))∥2 }
# ==============================

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19_relu())
        self.criterion = torch.nn.MSELoss()
        self.weights = [1.0/64, 1.0/64, 1.0/32, 1.0/32, 1.0/1]
        
        # InstanceNorm2d은 전체 tensor가 아니라 per channel로 Normalization을 진행함 (Layer norm과의 가장 큰 차이점)

        # duhyoenkim updated : dynamically allocate num_features
        # self.IN = nn.InstanceNorm2d(512, affine=False, track_running_stats=False)           # https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
        self.IN_layers = {
            'relu1_1': nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
            'relu2_1': nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
            'relu3_1': nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            'relu4_1': nn.InstanceNorm2d(512, affine=False, track_running_stats=False),
            'relu5_1': nn.InstanceNorm2d(512, affine=False, track_running_stats=False),
        }
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)                      # shape(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)                       # shape(1,3,1,1)
    
    def __call__(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)          # tensor.to(x) : x와 동일한 dtype + device로 mean(tensor)를 맞춤
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)             # output이 dictionary 형식으로 relu ouptut들을 return 함
        
        loss = 0
        for i, key in enumerate(['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']):
            IN_layer = self.IN_layers[key].to(x.device)
            loss += self.weights[i] * self.criterion(IN_layer(x_vgg[key]), IN_layer(y_vgg[key]))

        # loss  = self.weights[0] * self.criterion(self.IN(x_vgg['relu1_1']), self.IN(y_vgg['relu1_1']))
        # loss += self.weights[1] * self.criterion(self.IN(x_vgg['relu2_1']), self.IN(y_vgg['relu2_1']))
        # loss += self.weights[2] * self.criterion(self.IN(x_vgg['relu3_1']), self.IN(y_vgg['relu3_1']))
        # loss += self.weights[3] * self.criterion(self.IN(x_vgg['relu4_1']), self.IN(y_vgg['relu4_1']))
        # loss += self.weights[4] * self.criterion(self.IN(x_vgg['relu5_1']), self.IN(y_vgg['relu5_1']))

        return loss


class VGG19_relu(torch.nn.Module):
    def __init__(self):
        super(VGG19_relu, self).__init__()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        # duhyeonkim updated : ***the most important for "mps" -> mps doesnot support average pool2d in the vgg19 (.feature extracts only the Relu maxpool conv2d)
        cnn = models.vgg19(weights='IMAGENET1K_V1').features
        # cnn.load_state_dict(torch.load(os.path.join('./models/', 'vgg19-dcbb9e9d.pth')))
        cnn = cnn.to(self.device)
        
        # duhyeonkim updated (shorter code impl.)

        relu_layers = {
            'relu1_1': 1,
            'relu1_2': 3,
            'relu2_1': 6,
            'relu2_2': 8,
            'relu3_1': 11,
            'relu3_2': 13,
            'relu3_3': 15,
            'relu3_4': 17,
            'relu4_1': 20,
            'relu4_2': 22,
            'relu4_3': 24,
            'relu4_4': 26,
            'relu5_1': 29,
            'relu5_2': 31,
            'relu5_3': 33,
            'relu5_4': 35
        }

        self.layers = nn.ModuleDict()

        prev_idx = 0
        for name, idx in relu_layers.items():
            self.layers[name] = nn.Sequential(*list(cnn.children())[prev_idx:idx+1])
            prev_idx = idx + 1

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = {}
        for name, layer in self.layers.items():
            x = layer(x)
            outputs[name] = x
        return outputs


# ==============================
# (3) identity loss (exp to exp)
# Lidt = Exh ∼Ph [∥xh−G(xh)∥1]
# ==============================
class MultiscaleRecLoss(nn.Module):         # duhyeonkim updated : d2f에서는 수정이 필요해 보임
    def __init__(self, scale=3, rec_loss_type='l1', multiscale=True):
        super(MultiscaleRecLoss, self).__init__()
        self.multiscale = multiscale
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(rec_loss_type))
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)        # count_include_pad는 padding을 안해서 의미가 없긴한데
        if self.multiscale:
            self.weights = [1.0, 1.0/2, 1.0/4]
            self.weights = self.weights[:scale]

    def forward(self, input, target):
        loss = 0
        pred = input.clone()            # tensor는 =를 사용하면 mem addr를 공유하여 독립적이지 않음
        gt = target.clone()
        if self.multiscale:
            for i in range(len(self.weights)):
                loss += self.weights[i] * self.criterion(pred, gt)
                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            loss = self.criterion(pred, gt)
        return loss


# ==============================
# (1) Rahinge Adversarial loss 
# ==============================

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):           # tensor 바꿔야함
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        # self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'rahinge':
            pass
        elif gan_mode == 'rals':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        if self.gan_mode == 'original': # cross entropy loss
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(real_preds, target_tensor)
                return loss
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(fake_preds, target_tensor)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'ls':   
            if for_real:  
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                return F.mse_loss(real_preds, target_tensor)
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                return F.mse_loss(fake_preds, target_tensor)
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'hinge':
            if for_real:
                if for_discriminator:                
                    if target_is_real:
                        minval = torch.min(real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(real_preds)
                return loss
            elif for_fake:
                if for_discriminator:                
                    if target_is_real:
                        minval = torch.min(fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(fake_preds)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        
        elif self.gan_mode == 'rahinge':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)

                # generally, hinge loss forces to make D(x) >= 1, D(G(z)) <= -1
                loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))
                return loss / 2
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
                return loss / 2
        elif self.gan_mode == 'rals':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff - 1) ** 2) + torch.mean((f_r_diff + 1) ** 2)
                return loss / 2
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff + 1) ** 2) + torch.mean((f_r_diff - 1) ** 2)
                return loss / 2
        else:
            # wgan
            if for_real:
                if target_is_real:
                    return -real_preds.mean()
                else:
                    return real_preds.mean()
            elif for_fake:
                if target_is_real:
                    return -fake_preds.mean()
                else:
                    return fake_preds.mean()
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")

    def __call__(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        ## computing loss is a bit complicated because |input| may not be
        ## a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(real_preds, list):
            loss = 0
            for (pred_real_i, pred_fake_i) in zip(real_preds, fake_preds):
                if isinstance(pred_real_i, list):
                    pred_real_i = pred_real_i[-1]
                if isinstance(pred_fake_i, list):
                    pred_fake_i = pred_fake_i[-1]

                loss_tensor = self.loss(pred_real_i, pred_fake_i, target_is_real, for_real, for_fake, for_discriminator)        # return sclar

                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss
        else:
            return self.loss(real_preds, target_is_real, for_discriminator)

