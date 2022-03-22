import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


def create_loss_function(loss_config):
    if loss_config['label_smoothing'] == 0.0:
        if loss_config['name'] == 'ce':
            return nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0] * loss_config['true_class'] + [loss_config['na_weight']]))
        if loss_config['name'] == 'focal':
            return FocalLoss(gamma=loss_config['gamma'],
                             alpha=[loss_config['alpha']] * loss_config['true_class'] + [1 - loss_config['alpha']])
        if loss_config['name'] == 'ldam':
            return LDAMLoss(cls_num_list=loss_config['cls_num_list'],
                            max_m=loss_config['max_m'],
                            s=loss_config['s'])
        if loss_config['name'] == 'dice':
            return DiceLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma']
            )
        if loss_config['name'] == 'two':
            return TwoLoss(binary_weight=loss_config['na_weight'])
    else:
        if loss_config['name'] == 'ce':
            return LabelSmoothCrossEntropyLoss(weight=torch.FloatTensor([1.0] * loss_config['true_class'] + [loss_config['na_weight']]),
                                               alpha=loss_config['label_smoothing'])
        elif loss_config['name'] == 'focal':
            raise NotImplementedError
        elif loss_config['name'] == 'ldam':
            raise NotImplementedError
        elif loss_config['name'] == 'dice':
            raise NotImplementedError
        elif loss_config['name'] == 'two':
            raise NotImplementedError            
        
class TwoLoss(nn.Module):
    def __init__(self, binary_weight=1.0):
        super().__init__()
        self.binary_weight = binary_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        non_mask = targets != -100
        logits = logits[non_mask]
        targets = targets[non_mask]
        
        na_idx = logits.size(1) - 1
        
        na_logits = logits[:,-1]
        na_label = (targets == na_idx).float()
        na_loss = F.binary_cross_entropy_with_logits(na_logits, na_label, reduction="mean")
        
        label_logits = logits[:,:-1].log_softmax(1)
        na_label_mask = na_label == 0.0
        label_logits = label_logits[na_label_mask]
        label_targets = targets[na_label_mask]
        if na_label_mask.sum() > 0:
            label_loss = nn.CrossEntropyLoss(reduction="mean")(label_logits, label_targets)
        else:
            label_loss = 0.0
        loss = na_loss * self.binary_weight + label_loss
        return loss
        
        
class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.0):
        super().__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        if weight is not None:
            if isinstance(weight, list):
                weight = weight / sum(weight)
                self.weight = torch.FloatTensor(weight)
            else:
                self.weight = weight / weight.sum()
        else:
            self.weight = None
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        non_mask = targets != -100
        logits = logits[non_mask]
        targets = targets[non_mask]
        logits = logits.log_softmax(dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.alpha / (logits.size(1) - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
            
        if self.weight is None:
            return torch.mean(torch.sum(-true_dist * logits, dim=1))
        return torch.mean(torch.sum(-true_dist * logits * self.weight.to(logits.device).unsqueeze(0), dim=1)) # batch * class
        
        
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        # max_m should be tuning maybe 0.3~0.5
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        non_mask = target != -100
        x = x[non_mask]
        target = target[non_mask]
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :].to(x.device), index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha)
        self.reduction = reduction
        
    def forward(self, predicts, labels):
        class_size = predicts.size(1)
        labels = labels.reshape(-1).unsqueeze(1)
        predicts = predicts.transpose(1,0).reshape(class_size, -1).permute(1,0)
        non_mask = labels[:,0] != -100
        predicts = predicts[non_mask]
        labels = labels[non_mask]

        pt = F.softmax(predicts, dim=-1)
        logpt = F.log_softmax(predicts, dim=-1)
        pt = pt.gather(1, labels).squeeze(-1)
        logpt = logpt.gather(1, labels).squeeze(-1)
        
        if self.alpha is not None:
            at = self.alpha.to(predicts.device).gather(0, labels.squeeze(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
    

if __name__ == "__main__":
    loss_fn = TwoLoss()
    pred = torch.FloatTensor([[1.,2.,3.],[4.,-5.,-10.]])
    label = torch.LongTensor([0, 2])
    print(pred.shape, label.shape)
    print(loss_fn(pred, label))
    
