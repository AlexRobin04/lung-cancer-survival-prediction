import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.core_utils import *
#----> Attention module
class Attn_Net(nn.Module):
    
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 4):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

#----> Attention Gated module
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 256, D = 128, dropout = False, n_classes = 4):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.norm = nn.LayerNorm(D)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x) #[N, 256]
        b = self.attention_b(x) #[N, 256]
        A = a.mul(b) #torch.mul(a, b)
        A = self.norm(A)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class AMIL(nn.Module):
    def __init__(self, config, n_classes, gate=False, input_dim: int = 512):
        super(AMIL, self).__init__()
        # 支持不同特征维度（512/1024/…），首次 forward 自动推断 in_features
        fc = [nn.LazyLinear(256), nn.ReLU()]
        if gate:
            attention_net = Attn_Net_Gated(L = 256, D = 128, n_classes = 1)
        else:
            # 注意力应为单头（输出 Nx1），不是按类别数输出 NxC
            attention_net = Attn_Net(L = 256, D = 128, n_classes = 1)
        
        self.hard_or_soft = config.hard_or_soft
        self.loss_ce = nn.CrossEntropyLoss()
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(256, n_classes)

    def forward(
        self,
        x_s,
        coord_s=None,
        x_l=None,
        coords_l=None,
        label=None,
        staus=None,
        time=None,
        disc=None,
        soft_0=None,
        soft_1=None,
        soft_2=None,
        soft_3=None,
    ):
        
        h = x_s
        
        #---->Attention
        A, h = self.attention_net(h)  # NxK

        

        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N


        h = torch.mm(A, h) 

        #---->predict output
        logits = self.classifiers(h)  # [1, n_classes]
        logits = logits.view(1, -1)

        loss = None
        if (disc is not None) and (staus is not None):
            disc = disc.unsqueeze(1)
            staus = staus.unsqueeze(1)
            if self.hard_or_soft:
                loss = nll_loss_soft(
                    logits, disc, staus, soft_0, soft_1, soft_2, soft_3, alpha=0.4, eps=1e-7, reduction="mean"
                )
            else:
                loss = nll_loss(logits, disc, staus, alpha=0.4, eps=1e-7, reduction="mean")
        elif label is not None:
            loss = self.loss_ce(logits.view(1, -1), label.view(1))

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}

       

        return logits, Y_prob, loss













