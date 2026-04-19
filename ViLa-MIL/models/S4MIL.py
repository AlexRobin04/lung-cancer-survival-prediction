# This code is taken from the original S4 repository https://github.com/HazyResearch/state-spaces
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe
from utils.core_utils import *

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L)  # (H L)  % one dimensional Fourier transform of real-valued, signal len = 2*L
        u_f = torch.fft.rfft(u.to(torch.float32), n=2*L)  # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class S4Model(nn.Module):
    def __init__(self, config,in_dim, n_classes, dropout, act, d_model, d_state):
        super(S4Model, self).__init__()
        self.n_classes = n_classes
        # 支持不同特征维度（512/1024/…），首次 forward 自动推断 in_features
        self._fc1 = [nn.LazyLinear(d_model)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)
        self.s4_block = nn.Sequential(nn.LayerNorm(d_model),
                                      S4D(d_model=d_model, d_state=d_state, transposed=False))

        self.classifier = nn.Linear(d_model, self.n_classes)
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
        x = x.unsqueeze(0)

        x = self._fc1(x)
        x = self.s4_block(x)
        x = torch.max(x, axis=1).values
        # print(x.shape)
        logits = self.classifier(x)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None


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






    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.s4_block  = self.s4_block .to(device)
        self.classifier = self.classifier.to(device)
    
if __name__ == "__main__":

    x = torch.randn(50, 16)

    in_dim = 16           # 每个时间步的输入维度
    seq_len = 50          # 输入序列长度
    d_model = 32          # S4 隐藏维度
    d_state = 64          # 状态空间维度
    dropout = 0.1
    n_classes = 3
    act = "gelu"

    model = S4Model(in_dim=in_dim,
                    n_classes=n_classes,
                    dropout=dropout,
                    act=act,
                    d_model=d_model,
                    d_state=d_state)
    output = model(x)
    logits, Y_prob, Y_hat, _, _ = output
    print("Logits:", logits)
    print("Probabilities:", Y_prob)
    print("Predicted label:", Y_hat.item())
    
