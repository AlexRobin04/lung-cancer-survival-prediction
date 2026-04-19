import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
import sys
sys.path.append('/home/ubuntu/project/ViLa-MIL/')
from utils.core_utils import *
class WiKG(nn.Module):
    def __init__(self, config, dim_in=384, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn'):
        super().__init__()

        # 支持不同特征维度（512/1024/…），首次 forward 自动推断 in_features
        self._fc1 = nn.Sequential(nn.LazyLinear(dim_hidden), nn.LeakyReLU())
        
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError
        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)
        self.hard_or_soft = config.hard_or_soft
        self.loss_ce = nn.CrossEntropyLoss()
        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net=nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(), nn.Linear(dim_hidden//2, 1))     
            self.readout = GlobalAttention(att_net)


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
        x = x_s
        x = x.unsqueeze(0) 
        x = self._fc1(x)    # [B,N,C]

        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        
        h = self.message_dropout(embedding)

        h = self.readout(h.squeeze(0), batch=None)
        h = self.norm(h)
        logits = self.fc(h)


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

            
if __name__ == "__main__":
    data = torch.randn((1, 10000, 512)).cuda()
    model = WiKG(dim_in=512, dim_hidden=384, topk=6, n_classes=4, agg_type='bi-interaction', dropout=0.3, pool='attn').cuda()
    output = model(data)
    print(output.shape)