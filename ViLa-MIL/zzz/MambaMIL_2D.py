import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mamba_simple import MambaConfig as SimpleMambaConfig
from mamba_simple import Mamba as SimpleMamba

def split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MambaMIL_2D(nn.Module):
    def __init__(self,pos_emb_type):
        super(MambaMIL_2D, self).__init__()

        self._fc1 = [nn.Linear(512, 128)]
        self._fc1 += [nn.GELU()]
        self._fc1 += [nn.Dropout(0.25)]

        self._fc1 = nn.Sequential(*self._fc1)
        
        self.norm = nn.LayerNorm(128)
        
        self.layers = nn.ModuleList()
        self.patch_encoder_batch_size =128
        config = SimpleMambaConfig(
            d_model = 128,                      # 128
            n_layers = 1,                   # 1
            d_state = 16,                # 16
            inner_layernorms = True, # False or True
            pscan = True,                                # True
            use_cuda = False,                        # False
            mamba_2d = True,
            mamba_2d_max_w = 512,
            mamba_2d_max_h = 512,
            mamba_2d_pad_token = 'trainable',      # 'trainable'
            mamba_2d_patch_size = 512     #512
        )
        self.layers = SimpleMamba(config)
        self.config = config

        self.n_classes = 4
        

        self.attention = nn.Sequential(
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        self.classifier = nn.Linear(128, self.n_classes)
      

        if pos_emb_type == 'linear':
            self.pos_embs = nn.Linear(2, 128)
            self.norm_pe = nn.LayerNorm(128)
            self.pos_emb_dropout = nn.Dropout(0.0)
        else:
            self.pos_embs = None

        self.apply(initialize_weights)

    def forward(self, x, coords):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)   # (1, num_patch, feature_dim)
        h = x.float()  # [1, num_patch, feature_dim]

        h = self._fc1(h)  # [1, num_patch, mamba_dim];   project from feature_dim -> mamba_dim

        # Add Pos_emb
        # if self.pos_emb_type == 'linear':
        #     pos_embs = self.pos_embs(coords)
        #     h = h + pos_embs.unsqueeze(0)
        #     h = self.pos_emb_dropout(h)

        h = self.layers(h, coords, self.pos_embs)

        h = self.norm(h)   # LayerNorm
        A = self.attention(h) # [1, W, H, 1]

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:  
            A = A.permute(0,3,1,2)
            A = A.view(1,1,-1)
            h = h.view(1,-1,self.config.d_model)

        A = F.softmax(A, dim=-1)  # [1, 1, num_patch]  # A: attention weights of patches
        h = torch.bmm(A, h) # [1, 1, 512] , weighted combination to obtain slide feature
        h = h.squeeze(0)  # [1, 512], 512 is the slide dim

        logits = self.classifier(h)  # [1, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        results_dict = None



        return logits, Y_prob, Y_hat, results_dict # same return as other models
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


def generate_fake_data(num_patches=100, in_dim=384, max_coord=20):
    """
    生成模拟 patch-level 特征和坐标信息
    """
    features = torch.randn(num_patches, in_dim)  # (num_patches, in_dim)
    coords = torch.randint(0, max_coord, (num_patches, 2))  # (num_patches, 2)
    return features, coords



if __name__ == "__main__":
    x, coords = generate_fake_data(num_patches=2, in_dim=512,
                                   max_coord=max(16, 16))
    print(x)
    print(coords)

    pos_emb_type = 'linear'

    # 初始化模型
    model = MambaMIL_2D(pos_emb_type)
    model.eval()  # 测试阶段设为 eval 模式

    # 前向推理
    with torch.no_grad():
        logits, Y_prob, Y_hat, results_dict = model(x, coords)

    # 打印输出结果
    print("logits:", logits)
    print("Y_prob:", Y_prob)
    print("Y_hat:", Y_hat)