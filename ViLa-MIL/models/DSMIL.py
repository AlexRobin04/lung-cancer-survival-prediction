import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.core_utils import *

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        # 支持不同特征维度（512/1024/…），首次 forward 自动推断 in_features
        self.fc = nn.Sequential(nn.LazyLinear(out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.LazyLinear(128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.LazyLinear(input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self,config, in_size, num_class, dropout):
        super(MILNet, self).__init__()
        self.i_classifier = FCLayer(in_size, out_size=num_class)
        self.b_classifier = BClassifier(input_size=in_size, output_class=num_class, dropout_v=dropout)
        self.hard_or_soft = config.hard_or_soft
        self.loss_ce = nn.CrossEntropyLoss()
        
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

        feats, classes = self.i_classifier(x)
        logits, _, _ = self.b_classifier(feats, classes)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

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