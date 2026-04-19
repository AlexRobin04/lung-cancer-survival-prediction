
import torch
import torch.nn as nn
import sys
sys.path.append('/home/ubuntu/project/ViLa-MIL/') 
from modules.srmamba import SRMamba
from modules.bimamba import BiMamba
from modules.mamba_simple import Mamba
# from mamba_ssm import Mamba, SRMamba, BiMamba
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, n_classes=4, dropout=0.1, act='relu', n_features=1024, layer=2, rate=10, type="SRMamba"):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(n_features, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, data_s=None, coord_s=None, data_l=None, coords_l=None, label=None, **kwargs):
        # API training path: build expected packed tensor from two-scale features.
        if data_s is not None and data_l is not None and coord_s is not None:
            k = min(data_s.size(0), data_l.size(0), coord_s.size(0))
            h = torch.cat([data_s[:k], data_l[:k]], dim=1)  # [K, 1024]
            packed = torch.cat([h, coord_s[:k].float()], dim=1).unsqueeze(0)  # [1, K, 1026]
            h_all = [packed]
        else:
            h_all = kwargs['data']

        feature_patient = []
        # 0是超像素的坐标，1,2是patch的坐标，其余的为特征
        for h in h_all:  # All bags / slides corresponding to a patient
            h = h[:, :, :-2].clone().float()
            B, N, C = h.size()

            h = self._fc1(h)  # [B, n, 256]

            if self.type == "SRMamba":
                for layer in self.layers:
                    h_ = h
                    h = layer[0](h)
                    h = layer[1](h, rate=self.rate)
                    h = h + h_
            elif self.type == "Mamba" or self.type == "BiMamba":
                for layer in self.layers:
                    h_ = h
                    h = layer[0](h)
                    h = layer[1](h)
                    h = h + h_

            feature_patient.append(h)

        h = torch.cat(feature_patient, dim=1)
        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)
        # ---->predict
        logits = self.classifier(h)  # [B, n_classes]

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        # Keep backward compatibility while supporting unified training loop.
        if label is None:
            results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}
            return results_dict

        logits_for_ce = torch.logit(hazards.clamp(1e-6, 1 - 1e-6))
        Y_prob = F.softmax(logits_for_ce, dim=1)
        loss = F.cross_entropy(logits_for_ce, label.view(1))
        return logits_for_ce, Y_prob, loss
    
if __name__ == "__main__":
    data = torch.randn((1, 10000, 512)).cuda()
    model = MambaMIL(dn_classes=4, dropout=0.1, act='relu', n_features=1024, layer=2, rate=10, type="SRMamba").cuda()
    output = model(data)
    print(output.shape)