import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
# import torchsort
eps = 1e-8


class Loss(nn.modules.loss._Loss):
    """
    make loss function
    """
    def __init__(self, loss_str):
        super(Loss, self).__init__()

        self.loss = []
        for loss in loss_str.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'PLCCLoss':
                loss_function = PLCCLoss()
            elif loss_type == 'SRCCLoss':
                loss_function = SRCCLoss()
            elif loss_type == 'Rank':
                loss_function = RankHingedLoss()
            elif loss_type == 'Rela':
                loss_function = RelativeDistLoss()
            elif loss_type == 'norm-in-norm':
                loss_function = norm_loss_with_normalization()
            elif loss_type == 'min-max-norm':
                loss_function = norm_loss_with_min_max_normalization()
            elif loss_type == 'mean-norm':
                loss_function = norm_loss_with_mean_normalization()
            elif self.loss_type == 'scaling':
                loss_function = norm_loss_with_scaling()
            else:
                raise ValueError(f'Loss {loss_type} not supported !')

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

    def forward(self, input_mos, target_mos):
        loss_sum = 0.0
        loss_items = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](input_mos, target_mos)
                effective_loss = l['weight'] * loss
                loss_items[l['type']] = loss
                loss_sum += effective_loss
        loss_items['Total'] = loss_sum
        return loss_sum, loss_items


class RankHingedLoss(torch.nn.Module):
    def __init__(self, margin=0.05, y_margin=0.01):
        super(RankHingedLoss, self).__init__()
        self.margin = margin
        self.y_margin = y_margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 2
        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0

    def forward(self, x_pred, y_true):
        self.check_type_forward((x_pred, y_true))
        bs = y_true.shape[0]

        X_pred = x_pred.repeat(bs, 1)
        X_diff = X_pred - X_pred.t()
        Y_true = y_true.repeat(bs, 1)
        Y_diff = Y_true - Y_true.t()

        Y_diff[torch.abs(Y_diff) < self.y_margin] = 0.0
        Y_diff_sign = torch.sign(Y_diff)

        rank_diff = torch.clamp(self.margin - Y_diff_sign * X_diff, min=0.0)
        rank_diff = torch.triu(rank_diff, diagonal=1)
        rank_loss = torch.sum(rank_diff) / (bs * (bs - 1) / 2)
        return rank_loss


class RelativeDistLoss(nn.Module):
    def __init__(self, margin=0.05):
        super(RelativeDistLoss, self).__init__()
        self.margin = margin

    def forward(self, pred, label):
        b = len(pred)

        pred_matrix = pred.repeat(pred.shape[0], 1)
        pred_matrix_2 = pred_matrix.t()

        label_matrix = label.repeat(label.shape[0], 1)
        label_matrix_2 = label_matrix.t()

        pred_rank = pred_matrix - pred_matrix_2
        label_rank = label_matrix - label_matrix_2

        loss = torch.sum(torch.abs(pred_rank - label_rank)) / (2 * b)
        return loss


class PLCCLoss(nn.Module):
    def __init__(self):
        super(PLCCLoss, self).__init__()

    def forward(self, input, target):
        input0 = input - torch.mean(input)
        target0 = target - torch.mean(target)
        loss = torch.sum(input0 * target0) / ((torch.sqrt(torch.sum(input0 ** 2))
                                                   * torch.sqrt(torch.sum(target0 ** 2))) + eps)
        return 1-torch.abs(loss)



class linearity_induced_loss(nn.Module):
    def __init__(self, alpha=[1, 1]):
        super(linearity_induced_loss, self).__init__()
        self.alpha = alpha
    def forward(self, y_pred, y, detach=False):
        """linearity-induced loss, actually MSE loss with z-score normalization"""
        if y_pred.size(0) > 1:  # z-score normalization: (x-m(x))/sigma(x).
            sigma_hat, m_hat = torch.std_mean(y_pred.detach(), unbiased=False) if detach else torch.std_mean(y_pred, unbiased=False)
            y_pred = (y_pred - m_hat) / (sigma_hat + eps)
            sigma, m = torch.std_mean(y, unbiased=False)
            y = (y - m) / (sigma + eps)
            scale = 4
            loss0, loss1 = 0, 0
            if self.alpha[0] > 0:
                loss0 = F.mse_loss(y_pred, y) / scale  # ~ 1 - rho, rho is PLCC
            if self.alpha[1] > 0:
                rho = torch.mean(y_pred * y)
                loss1 = F.mse_loss(rho * y_pred, y) / scale  # 1 - rho ** 2 = 1 - R^2, R^2 is Coefficient of determination
            
            return (self.alpha[0] * loss0 + self.alpha[1] * loss1) / (self.alpha[0] + self.alpha[1])
        else:
            return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.

class norm_loss_with_normalization(nn.Module):
    def __init__(self):
        super(norm_loss_with_normalization, self).__init__()

        
    def forward(self, y_pred, y, alpha=[1, 1], p=1, q=2, detach=False, exponent=True):
        """norm_loss_with_normalization: norm-in-norm"""
        N = y_pred.size(0)
        if N > 1:  
            m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
            y_pred = y_pred - m_hat  # very important!!
            normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)  # Actually, z-score normalization is related to q = 2.
            # print('bhat = {}'.format(normalization.item()))
            y_pred = y_pred / (eps + normalization)  # very important!
            y = y - torch.mean(y)
            y = y / (eps + torch.norm(y, p=q))
            scale = np.power(2, max(1,1./q)) * np.power(N, max(0,1./p-1./q)) # p, q>0
            loss0, loss1 = 0, 0
            if alpha[0] > 0:
                err = y_pred - y
                if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                    err += eps 
                loss0 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to PLCC
                loss0 = torch.pow(loss0, p) if exponent else loss0 #
            if alpha[1] > 0:
                rho =  torch.cosine_similarity(y_pred.t(), y.t())  #  
                err = rho * y_pred - y
                if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                    err += eps 
                loss1 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to LSR
                loss1 = torch.pow(loss1, p) if exponent else loss1 #  #  
            return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
        else:
            return F.l1_loss(y_pred, y)  # 0 for batch with single sample.

class norm_loss_with_min_max_normalization(nn.Module):
    def __init__(self):
        super(norm_loss_with_min_max_normalization, self).__init__()
    def forward(self, y_pred, y, alpha=[1, 1], detach=False):
        if y_pred.size(0) > 1:  
            m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
            M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
            y_pred = (y_pred - m_hat) / (eps + M_hat - m_hat)  # min-max normalization
            y = (y - torch.min(y)) / (eps + torch.max(y) - torch.min(y))
            loss0, loss1 = 0, 0
            if alpha[0] > 0:
                loss0 = F.mse_loss(y_pred, y)
            if alpha[1] > 0:
                rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
                loss1 = F.mse_loss(rho * y_pred, y) 
            return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
        else:
            return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.

class norm_loss_with_mean_normalization(nn.Module):
    def __init__(self, alpha=[1, 1]):
        super(norm_loss_with_mean_normalization, self).__init__()

    def forward(self, y_pred, y, alpha=[1, 1], detach=False):
        if y_pred.size(0) > 1:  
            mean_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
            m_hat = torch.min(y_pred.detach()) if detach else torch.min(y_pred)
            M_hat = torch.max(y_pred.detach()) if detach else torch.max(y_pred)
            y_pred = (y_pred - mean_hat) / (eps + M_hat - m_hat)  # mean normalization
            y = (y - torch.mean(y)) / (eps + torch.max(y) - torch.min(y))
            loss0, loss1 = 0, 0
            if alpha[0] > 0:
                loss0 = F.mse_loss(y_pred, y) / 4
            if alpha[1] > 0:
                rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
                loss1 = F.mse_loss(rho * y_pred, y) / 4
            return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
        else:
            return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.

class norm_loss_with_scaling(nn.Module):
    def __init__(self):
        super(norm_loss_with_scaling, self).__init__()
    def forward(self, y_pred, y, alpha=[1, 1], p=2, detach=False):
        if y_pred.size(0) > 1:  
            normalization = torch.norm(y_pred.detach(), p=p) if detach else torch.norm(y_pred, p=p) 
            y_pred = y_pred / (eps + normalization)  # mean normalization
            y = y / (eps + torch.norm(y, p=p))
            loss0, loss1 = 0, 0
            if alpha[0] > 0:
                loss0 = F.mse_loss(y_pred, y) / 4
            if alpha[1] > 0:
                rho =  torch.cosine_similarity(y_pred.t(), y.t())  #
                loss1 = F.mse_loss(rho * y_pred, y) / 4
            return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
        else:
            return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.

class SRCCLoss(nn.Module):
    def __init__(self):
        super(SRCCLoss, self).__init__()

    def forward(self, pred, target, **kw):
        b, n = pred.shape
        pred = pred.view(n, b)
        target = target.view(n, b)
        pred = torchsort.soft_rank(pred, **kw)
        target = torchsort.soft_rank(target, **kw)
        pred = pred - pred.mean()
        pred = pred / (pred.norm()+eps)
        target = target - target.mean()
        target = target / (target.norm()+eps)
        return 1 - (pred * target).sum()

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torch.autograd import Variable
    ref = Variable(torch.rand(8, 1)).to(device) # b, c, n, h, w
    dist = Variable(torch.rand(8, 1)).to(device) # b, c, n, h, w    
    print(spearmanr(ref, dist))
    # loss = SRCCLoss()
    # print(loss(ref, dist))