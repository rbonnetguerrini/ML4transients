import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StandardLoss:
    """Standard binary cross entropy loss"""
    
    def __init__(self):
        self.name = "standard"
    
    def __call__(self, y_pred, y_true, **kwargs):
        return F.binary_cross_entropy_with_logits(y_pred.squeeze(), y_true.float())


class CoTeachingLoss:
    """Co-teaching loss for binary classification"""
    
    def __init__(self, forget_rate_0=0.015, forget_rate_1=0.005):
        self.forget_rate_0 = forget_rate_0
        self.forget_rate_1 = forget_rate_1
        self.name = "coteaching"
    
    def __call__(self, y_1, y_2, t, epoch_forget_rates=None, **kwargs):
        if epoch_forget_rates is not None:
            forget_rate_0, forget_rate_1 = epoch_forget_rates
        else:
            forget_rate_0, forget_rate_1 = self.forget_rate_0, self.forget_rate_1
            
        return self._loss_coteaching_binary_asym(y_1, y_2, t, forget_rate_0, forget_rate_1)
    
    def _loss_coteaching_binary_asym(self, y_1, y_2, t, forget_rate_0, forget_rate_1):
        t = t.float()
        
        loss_1 = F.binary_cross_entropy_with_logits(y_1.squeeze(), t, reduction='none')
        loss_2 = F.binary_cross_entropy_with_logits(y_2.squeeze(), t, reduction='none')
        
        ind_1_sorted = torch.argsort(loss_1.data)
        ind_2_sorted = torch.argsort(loss_2.data)
        
        ind_class_0 = (t == 0).nonzero(as_tuple=True)[0]
        ind_class_1 = (t == 1).nonzero(as_tuple=True)[0]
        
        ind_1_class_0_sorted = ind_1_sorted[torch.isin(ind_1_sorted, ind_class_0)]
        ind_1_class_1_sorted = ind_1_sorted[torch.isin(ind_1_sorted, ind_class_1)]
        ind_2_class_0_sorted = ind_2_sorted[torch.isin(ind_2_sorted, ind_class_0)]
        ind_2_class_1_sorted = ind_2_sorted[torch.isin(ind_2_sorted, ind_class_1)]
        
        remember_rate_0 = 1 - forget_rate_0
        remember_rate_1 = 1 - forget_rate_1
        num_remember_0 = int(remember_rate_0 * len(ind_class_0))
        num_remember_1 = int(remember_rate_1 * len(ind_class_1))
        
        ind_1_update = torch.cat((ind_1_class_0_sorted[:num_remember_0], 
                                ind_1_class_1_sorted[:num_remember_1]))
        ind_2_update = torch.cat((ind_2_class_0_sorted[:num_remember_0], 
                                ind_2_class_1_sorted[:num_remember_1]))
        
        loss_1_update = F.binary_cross_entropy_with_logits(
            y_1[ind_2_update].squeeze(), t[ind_2_update], reduction='none')
        loss_2_update = F.binary_cross_entropy_with_logits(
            y_2[ind_1_update].squeeze(), t[ind_1_update], reduction='none')
        
        return torch.mean(loss_1_update), torch.mean(loss_2_update)


def get_loss_function(loss_type, **kwargs):
    """Factory function to get loss function"""
    if loss_type == "standard":
        return StandardLoss()
    elif loss_type == "coteaching":
        return CoTeachingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")