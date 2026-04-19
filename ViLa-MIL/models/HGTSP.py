import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import Mlp
from einops import rearrange, reduce
from torch import nn, einsum
from torch.nn import Linear, LayerNorm, ReLU
from timm.models.layers import trunc_normal_
import math
import nmslib
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch import nn, einsum
from torch_geometric.data import Data as geomData
from itertools import chain

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class RegionalAttention(nn.Module):

    def __init__(self, dim=512, region_size=49, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.region_size = region_size
        self.shift = int(np.sqrt(region_size))
        self.num_heads = 8
        head_dim = dim // 8
        self.scale = head_dim ** -0.5

        # ---->Attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        dim = 512
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, dim, kernel_size=1)
        )

    def forward(self, x):
        B, N, C = x.shape  # [b, n, c]

        # ---->partition regions
        x = rearrange(x, 'b (w ws) c -> b w ws c', ws=self.region_size)
        x1 = rearrange(x, 'b w ws c -> b c w ws', ws=self.region_size)
        x1 = self.spatial_interaction(x1)
        x1 = rearrange(x1, 'b c w ws -> b w ws c', ws=self.region_size)
        x = x + x1
        x = rearrange(x, 'b w ws c -> (b w) ws c')  # [b*num_region, region, C]

        # ---->attention
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # [b*num_region, num_head, region, C//num_head]

        q = q * self.scale  # Q: [b*num_region, num_head, region, C//num_head] K: [b*num_region, num_head , C//num_head, region]
        attn = (q @ k.transpose(-2, -1))  # [num_region, num_head, region, region]

        attn = attn  # + relative_position_bias
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v)

        x = out.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # ---->region_reverse
        x = rearrange(x, '(b w) ws c -> b (w ws) c', b=B)
        return x

class SimTransformer(nn.Module):
    def __init__(self, in_dim, proj_qk_dim=None, proj_v_dim=None, epsilon=None):
        """
        in_dim: the dimension of input.
        proj_qk_dim: the dimension of projected Q, K.
        proj_v_dim: the dimension of projected V.
        topk: number of patches with highest attention values.
        """
        super(SimTransformer, self).__init__()
        self._markoff_value = 0
        self.epsilon = epsilon
        if proj_qk_dim is None:
            proj_qk_dim = in_dim
        if proj_v_dim is None:
            proj_v_dim = in_dim
        self.proj_qk = nn.Linear(in_dim, proj_qk_dim, bias=False)
        nn.init.xavier_uniform_(self.proj_qk.weight)
        self.proj_v = nn.Linear(in_dim, proj_v_dim, bias=False)
        nn.init.xavier_uniform_(self.proj_v.weight)
        self.norm = nn.LayerNorm(proj_v_dim)

    def forward(self, x):
        q, k, v = self.proj_qk(x), self.proj_qk(x), self.proj_v(x)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        attention = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        if self.epsilon is not None:
            mask = (attention > self.epsilon).detach().float()
            attention = attention * mask + self._markoff_value * (1 - mask)
        out = torch.matmul(attention, v)
        out = self.norm(out)
        return out


class RegionalAttentionBlock(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, region_size=49):
        super().__init__()
        self.region_size = region_size
        self.norm1 = norm_layer(dim)
        self.RA = RegionalAttention(dim=dim, region_size=region_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        # ---->region
        x_out = self.RA(self.norm1(x))
        x = x + x_out
        x = self.act(x)
        x = self.drop(x)
        return x


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=False):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


class Cross_Attention(nn.Module):
    def __init__(self, global_dim=512, context_dim=512, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = 512
        print(inner_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(global_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, global_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, context):  # self.CA(self.prototype, h)
        ## x1:global_token, context:
        h = self.heads

        q = self.to_q(x1)
        k, v = self.to_k(context), self.to_v(context)
        #         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        #         print(q.shape, k.shape, v.shape)
        sim_ori = einsum('b i d, b j d -> b i j', q,
                         k) * self.scale  # [1,1,d] [1, num_patches, d] -- > [1, 1, num_patches]

        _attn = sim_ori.softmax(dim=-1)
        attn_score = _attn.squeeze(0)

        #         x = rearrange(x, 'b (ws w) c -> b w ws c', ws=self.region_size)
        attn_score = rearrange(attn_score, 'b (num_region ws) -> b num_region ws', ws=49)
        attn_score = torch.mean(attn_score, dim=2)

        attn = self.dropout(_attn)
        out = einsum('b i j, b j d -> b i d', attn, v)

        return self.to_out(out), attn_score, _attn


class HGTSP(nn.Module):
    def __init__(self, dropout=False, n_classes=4, region_size=49, topk=70, RegionK=9):
        super(HGTSP, self).__init__()
        self.n_classes = n_classes
        self.topk = topk
        self.region_size = region_size

        self.prototype = torch.nn.Parameter(torch.randn((1, 1, 512), requires_grad=True))  # , dtype=torch.float16
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.CA = Cross_Attention()
        self.RAB = RegionalAttentionBlock(dim=512, region_size=region_size)

        self.layers = torch.nn.ModuleList()
        self.num_layers = 1
        for i in range(1, self.num_layers + 1):
            conv = GENConv(512, 512, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(512, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.SRA = SimTransformer(512, proj_qk_dim=512, proj_v_dim=512)
        self.mlp1 = Mlp(in_features=512, hidden_features=512, act_layer=nn.GELU, drop=0.1)

        self.norm0 = nn.LayerNorm(512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.GAP = Attn_Net(L=512, D=256, dropout=dropout, n_classes=1)
        self._fc2 = nn.Linear(512, self.n_classes)

    def pt2graph(self, features, coords, radius=9):
        features = features.squeeze(0).cpu().detach().numpy()
        coords = coords.squeeze(0).cpu().detach().numpy()
        num_patches = features.shape[0]  # shape[0] 或者shape[1]

        model = Hnsw(space='l2')
        model.fit(features)
        a = np.repeat(range(num_patches), radius - 1)
        b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),
                        dtype=int)
        edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        G = geomData(x=torch.Tensor(features),
                     edge_latent=edge_latent,
                     centroid=torch.Tensor(coords))

        return G

    def forward(self, **kwargs):
        h_all = kwargs['data']  # list [[B,n,1024],...,[],[]]

        feature_patient = []
        for h in h_all:  # All WSIs corresponding to a patient

            # ---->Separate feature, coords information
            coords = h[:, :, :2].clone()  # [n, 2]
            h = h[:, :, 2:]  # [n, 1024]

            # ---->Dimensionality reduction
            h = self._fc1(h)  # [B, n, 512]

            # ---->cross attention
            prototype, attn_score, attn = self.CA(self.prototype, h)  # prototype: [1, 1, 512] attn_score: [1, num_regions] attn:[1, 1, num_patches]

            # ---->topk
            feature = rearrange(h, 'b (w ws) c -> b w ws c', ws=self.region_size)
            feature = rearrange(feature, 'b w ws c -> (b w) ws c', ws=self.region_size)
            coords = rearrange(coords, 'b (w ws) c -> b w ws c', ws=self.region_size)
            coords = rearrange(coords, 'b w ws c -> (b w) ws c', ws=self.region_size)

            topk = min(self.topk, feature.shape[0])
            topk_indices = torch.topk(attn_score, topk, dim=1)[1]
            feature = feature[topk_indices.squeeze(0), :, :]
            coords = coords[topk_indices.squeeze(0), :, :]


            feature = rearrange(feature, '(b w) ws c -> b (w ws) c', b=1)
            coords = rearrange(coords, '(b w) ws c -> b (w ws) c', b=1)

            # ---->RTB
            feature = self.norm0(self.RAB(feature))

            feature = rearrange(feature, 'b (w ws) c -> b w ws c', ws=self.region_size)
            feature = rearrange(feature, 'b w ws c -> (b w) ws c', ws=self.region_size)

            coords = rearrange(coords, 'b (w ws) c -> b w ws c', ws=self.region_size)
            coords = rearrange(coords, 'b w ws c -> (b w) ws c', ws=self.region_size)

            # 1. avg
            feature = torch.mean(feature, dim=1)
            coords = torch.mean(coords, dim=1)
            feature = rearrange(feature, '(b w) c -> b w c', b=1)
            coords = rearrange(coords, '(b w) c -> b w c', b=1)

            # # 2. max attention
            # num_region = feature.shape[0]
            # max_indices = torch.argmax(attn, dim=1)
            # feature = feature[torch.arange(num_region), max_indices]
            # coords = coords[torch.arange(num_region), max_indices]
            # feature = rearrange(feature, '(b w) c -> b w c', b=1)
            # coords = rearrange(coords, '(b w) c -> b w c', b=1)


            # 1. 通过RAB之后的特征进行构图, 填充
            r = 9
            if feature.shape[1] < r:
                feature = feature.repeat(1, r // feature.shape[1] + 1, 1)[:, :r, :]
                coords = coords.repeat(1, r // coords.shape[1] + 1, 1)[:, :r, :]
            G = self.pt2graph(feature, coords, radius=r).to('cuda')
            edge_index = G.edge_latent
            edge_attr = None

            gcn_f0 = self.layers[0].conv(feature, edge_index, edge_attr)
            gcn_final_out = torch.cat([feature, gcn_f0], axis=1)
            for layer in self.layers[1:]:
                gcn_f = layer(feature, edge_index, edge_attr)
                gcn_final_out = torch.cat([gcn_final_out, gcn_f], axis=1)
            gcn_final_out = rearrange(gcn_final_out, 'b (layer_num d) c -> b layer_num d c',
                                      layer_num=self.num_layers + 1)
            gcn_final_out = torch.mean(gcn_final_out, dim=1)
            feature = (feature + gcn_final_out) / 2

            # ---->shuffleregion
            feature = torch.cat([feature, prototype], dim=1)

            feature = feature + self.SRA(self.norm1(feature))
            feature = feature + self.mlp1(self.norm2(feature))

            feature_patient.append(feature)
        # ---->concat
        feature = torch.cat(feature_patient, dim=1)

        # ---->patient-level attention
        feature = self.norm3(feature)  # B, N, C
        A, feature = self.GAP(feature.squeeze(0))  # B C 1
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        feature = torch.mm(A, feature)

        # ---->predict output
        logits = self._fc2(feature)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}

        return results_dict

    def forward_vis(self, **kwargs):
        h_all = kwargs['data']  # list [[B,n,1024],...,[],[]]

        feature_patient = []
        h = h_all[0]  # [B,n,1024]

        # ---->Dimensionality reduction
        h = self._fc1(h)  # [B, n, 512]

        # ---->cross attention
        prototype, attn_score, attn = self.CA(self.prototype,
                                              h)  # prototype: [1, 1, 512] attn_score: [1, num_regions]  attn:[1, 1, num_patches]

        return attn_score, attn
