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
    """Original Co-teaching loss for binary classification
    
    Uses a single forget rate for all samples regardless of class.
    Based on: Han et al. (2018) - Co-teaching: Robust training of deep neural 
    networks with extremely noisy labels.
    
    Args:
        forget_rate: Fraction of samples to reject (default: 0.2)
    """
    
    def __init__(self, forget_rate=0.2):
        self.forget_rate = forget_rate
        self.name = "coteaching"
    
    def __call__(self, y_1, y_2, t, epoch_forget_rate=None, **kwargs):
        """Compute co-teaching loss
        
        Args:
            y_1: Predictions from network 1
            y_2: Predictions from network 2
            t: Ground truth labels
            epoch_forget_rate: Optional forget rate for this epoch (overrides default)
            
        Returns:
            Tuple of (loss_1, loss_2)
        """
        forget_rate = epoch_forget_rate if epoch_forget_rate is not None else self.forget_rate
        return self._loss_coteaching_binary(y_1, y_2, t, forget_rate)
    
    def _loss_coteaching_binary(self, y_1, y_2, t, forget_rate):
        """Original co-teaching with single forget rate"""
        t = t.float()
        
        # Compute per-sample losses
        loss_1 = F.binary_cross_entropy_with_logits(y_1.squeeze(), t, reduction='none')
        loss_2 = F.binary_cross_entropy_with_logits(y_2.squeeze(), t, reduction='none')
        
        # Calculate number of samples to keep
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(t))
        
        # Sort samples by loss (ascending)
        ind_1_sorted = torch.argsort(loss_1.data)
        ind_2_sorted = torch.argsort(loss_2.data)
        
        # Keep samples with smallest losses
        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        
        # Network 1 trained on samples selected by Network 2
        loss_1_update = F.binary_cross_entropy_with_logits(
            y_1[ind_2_update].squeeze(), t[ind_2_update], reduction='none')
        
        # Network 2 trained on samples selected by Network 1
        loss_2_update = F.binary_cross_entropy_with_logits(
            y_2[ind_1_update].squeeze(), t[ind_1_update], reduction='none')
        
        return torch.mean(loss_1_update), torch.mean(loss_2_update)


class CoTeachingAsymLoss:
    """Asymmetric Co-teaching loss for binary classification
    
    Uses different forget rates for different classes to handle class imbalance
    or class-specific label noise.
    
    Args:
        forget_rate_0: Forget rate for class 0 (default: 0.015)
        forget_rate_1: Forget rate for class 1 (default: 0.005)
    """
    
    def __init__(self, forget_rate_0=0.015, forget_rate_1=0.005):
        self.forget_rate_0 = forget_rate_0
        self.forget_rate_1 = forget_rate_1
        self.name = "coteaching_asym"
    
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


class StochasticCoTeachingLoss:
    """Stochastic co-teaching loss for binary classification
    
    Based on: Jansen et al. (2023) - Stochastic co-teaching for training neural 
    networks with unknown levels of label noise.
    https://www.nature.com/articles/s41598-023-43864-7
    
    Args:
        alpha: Beta distribution alpha parameter (recommended: 32)
        beta: Beta distribution beta parameter (recommended: 2-4)
        max_iters: Maximum number of iterations (typically epochs)
        tp_gradual: Number of iterations for gradual warmup
        delay: Number of iterations before starting rejection (default: 0)
        exponent: Exponent for schedule (default: 1)
        clip: Tuple of (min, max) values to clip random thresholds (default: (0.01, 0.99))
        seed: Random seed for reproducibility (default: 808)
    """
    
    def __init__(self, alpha=32, beta=4, max_iters=200, tp_gradual=10, 
                 delay=0, exponent=1, clip=(0.01, 0.99), seed=808):
        self.alpha = alpha
        self.beta = beta
        self.tp_gradual = tp_gradual
        self.clip = clip
        self.name = "stochastic_coteaching"
        
        # Create rate schedule: controls when stochastic rejection is active
        maxval = 1
        rate_schedule = np.ones(max_iters) * maxval
        rate_schedule[:delay] = 0  # No rejection during delay period
        rate_schedule[delay:delay+tp_gradual] = np.linspace(0, maxval ** exponent, tp_gradual)
        self.rate_schedule = rate_schedule
        
        # Initialize iteration counter and random number generator
        self._it = 0
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Loss function (binary cross entropy, no reduction)
        self.loss_fn = lambda logits, targets: F.binary_cross_entropy_with_logits(
            logits.squeeze(), targets.float(), reduction='none'
        )
        
        # Track rejection statistics
        self.fraction_reject_1 = 0.0
        self.fraction_reject_2 = 0.0
        self.probas1 = None
        self.probas2 = None
    
    def step(self):
        """Increment iteration counter (call after each epoch)"""
        self._it += 1
    
    def get_probas(self, logits, y):
        """Get predicted probabilities of the ground-truth class
        
        For binary classification, this extracts the probability that the model
        assigns to the class indicated by y (0 or 1).
        
        Args:
            logits: Model outputs (before sigmoid)
            y: Ground truth labels (0 or 1)
            
        Returns:
            Probabilities of the ground-truth class
        """
        # Get probabilities for both classes
        probs_class_1 = torch.sigmoid(logits.squeeze())
        probs_class_0 = 1 - probs_class_1
        
        # Stack them: shape (batch_size, 2)
        all_probabilities = torch.stack([probs_class_0, probs_class_1], dim=1)
        
        # Gather probabilities corresponding to ground truth labels
        class_probabilities = torch.gather(all_probabilities, 1, y.long().unsqueeze(1))
        
        return class_probabilities.squeeze()
    
    def mask_probas(self, probas):
        """Create binary mask by comparing probabilities to random Beta thresholds
        
        Args:
            probas: Predicted probabilities of ground-truth class
            
        Returns:
            Binary mask (1 = keep sample, 0 = reject sample)
        """
        maxval = self.rate_schedule[self._it]
        
        # During delay period, keep all samples
        mask = torch.ones_like(probas)
        if maxval == 0:
            return mask
        
        # Sample random thresholds from Beta distribution
        rand = torch.from_numpy(
            maxval * self.rng.beta(a=self.alpha, b=self.beta, size=probas.shape).astype(np.float32)
        ).to(probas.device)
        
        # Clip thresholds to avoid extreme values
        rand = torch.clamp(rand, *self.clip)
        
        # Keep samples where probability > random threshold
        mask = torch.gt(probas, rand).float()
        
        return mask
    
    @torch.no_grad()
    def current_fraction_rejected(self):
        """Get current fraction of rejected samples for both networks"""
        return self.fraction_reject_1, self.fraction_reject_2
    
    @torch.no_grad()
    def current_probas(self):
        """Get current probabilities for both networks"""
        return self.probas1, self.probas2
    
    def __call__(self, logits_1, logits_2, y, **kwargs):
        """Compute stochastic co-teaching loss
        
        Args:
            logits_1: Predictions from network 1
            logits_2: Predictions from network 2
            y: Ground truth labels
            
        Returns:
            Tuple of (loss_1, loss_2) where each network is trained on samples
            selected by the other network
        """
        with torch.no_grad():
            # Get probabilities of ground-truth class for both networks
            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)
            
            # Create masks using stochastic thresholds
            # Retry if rejection rate is too high (>90%)
            loop = 0
            while True:
                mask_1 = self.mask_probas(self.probas1)
                mask_2 = self.mask_probas(self.probas2)
                
                # Check that at least 10% of samples are kept
                if (mask_1.sum() / mask_1.numel()) >= 0.1 and \
                   (mask_2.sum() / mask_2.numel()) >= 0.1:
                    break
                
                if loop == 5:
                    raise RuntimeError('More than 90 percent consistently rejected')
                
                loop += 1
            
            # Track rejection statistics
            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))
        
        # Compute losses
        loss_1 = self.loss_fn(logits_1, y)
        loss_2 = self.loss_fn(logits_2, y)
        
        # Network 1 trained on samples selected by Network 2
        loss_1_update = torch.sum(loss_1 * mask_2) / mask_2.sum()
        
        # Network 2 trained on samples selected by Network 1
        loss_2_update = torch.sum(loss_2 * mask_1) / mask_1.sum()
        
        return loss_1_update, loss_2_update


def get_loss_function(loss_type, **kwargs):
    """Factory function to get loss function"""
    if loss_type == "standard":
        return StandardLoss()
    elif loss_type == "coteaching":
        return CoTeachingLoss(**kwargs)
    elif loss_type == "coteaching_asym":
        return CoTeachingAsymLoss(**kwargs)
    elif loss_type == "stochastic_coteaching":
        return StochasticCoTeachingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")