from numpy import deprecate
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# kl_loss = nn.KLDivLoss(reduction="batchmean")
# ce_loss = torch.nn.CrossEntropyLoss()

def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()  # dim=-1
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def softmax_sigmoid_BCEloss(pred, targets):
    prob = torch.nn.functional.softmax(pred, dim=1)
    prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
    logit = torch.log((prob / (1 - prob)))
    loss_func = torch.nn.BCEWithLogitsLoss()
    return loss_func(logit, targets)

def norm_logits_BCEloss(pred, targets):
    loss_func = torch.nn.BCEWithLogitsLoss()
    return loss_func(pred, targets)

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1, #0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    support soft label
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)
    p_t = torch.abs(targets - p)
    loss = ce_loss * (p_t ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@deprecate
def sigmoid_ASL_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    gamma_: float = 2,
    c: float = 0.05,
    reduction: str = "mean",
):
    """
    NOT support soft label
    """
    p = torch.sigmoid(inputs)
    neg_flag = (1 - targets).float()
    p = torch.clamp(p - neg_flag * c, 1e-9)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_pos = ((1 - p)** gamma) * targets + (p** gamma_) * (1 - targets)
    loss = ce_loss * p_pos

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def ranking_loss(y_pred, y_true, scale_ = 2.0, margin_ = 1):
    y_pred *= scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)


class AsymmetricLoss_partial(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss_partial, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, thresh_pos=0.9, thresh_neg=-0.9, if_partial=True):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        y_pos = (y > thresh_pos).float()
        y_neg = (y < thresh_neg).float()
        # Basic CE calculation
        los_pos = y_pos * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = y_neg * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y_pos
            pt1 = xs_neg * y_neg  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_pos + self.gamma_neg * y_neg
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / x.shape[0] if if_partial else -loss.mean()

def dualcoop_loss(inputs, inputs_g, targets):
    """
    using official ASL loss.
    """
    loss_fun = AsymmetricLoss_partial(gamma_neg=2, gamma_pos=1, clip=0.05)

    return loss_fun(inputs, targets, thresh_pos=0.9, thresh_neg=-0.9) # + loss_fun(inputs_g, targets)


def ASL_loss(inputs, targets):
    """
    full label ASLOSS
    """
    loss_fun = AsymmetricLoss_partial(gamma_neg=2, gamma_pos=1, clip=0.05)

    return loss_fun(inputs, targets, thresh_pos=0.9, thresh_neg=0.9, if_partial=False)


import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, n_cls, bias=False, prob_path=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))

        self.n_cls = n_cls
        A = torch.eye(n_cls).float() * (1 - 0.001 * self.n_cls)
        A += 0.001
        self.A = nn.Parameter(A)
           
        # A = torch.ones(n_cls, n_cls).float() * (1 - 0.001 * self.n_cls)
        # A += 0.001
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj=None):
        support = torch.matmul(input, self.weight)
        if adj is None:
            adj = self.A
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_module(nn.Module):
    def __init__(self, layers=1, init_prob=False, init_prob_file=''):
        super(GC_module, self).__init__()
        print("Init GC_module")
        self.count_prob = torch.load(init_prob_file) 
        n_cls = self.count_prob.shape[0]
        
        self.layer_num = layers
        if self.layer_num == 1:
            self.gc1 = GraphConvolution(1024, 1024, n_cls)
        elif self.layer_num == 2:
            self.gc1 = GraphConvolution(1024, 1024, n_cls)
            self.relu = nn.LeakyReLU(0.15)
            self.gc2 = GraphConvolution(1024, 1024, n_cls)
        
        if init_prob:
            t = 0.3
            self.count_prob[self.count_prob < t] = 0
            # self.count_prob[self.count_prob >= t] = 1
            
            # p = 0.8
            # _adj_other = (1-p) / (_adj.sum(1, keepdims=True) + 1)
            # _adj = _adj * _adj_other
            # _adj = _adj + torch.eye(n_cls) * (p - _adj_other)
            # self.adj = _adj
            
            self.adj = nn.Parameter(self.count_prob)
        else:
            self.adj = None

    def forward(self, input, adj=None):
        if self.layer_num == 1:
            x = self.gc1(input, self.adj)
        elif self.layer_num == 2:
            x = self.gc1(input, self.adj)
            x = self.relu(x)
            x = self.gc2(x, self.adj)
        return x

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    # def gen_A(self, num_classes, t, adj_file):
    #     import pickle
    #     result = pickle.load(open(adj_file, 'rb'))
    #     _adj = result['adj']
    #     _nums = result['nums']
    #     _nums = _nums[:, np.newaxis]
    #     _adj = _adj / _nums
    #     _adj[_adj < t] = 0
    #     _adj[_adj >= t] = 1
    #     _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    #     _adj = _adj + np.identity(num_classes, np.int)
    #     return _adj
