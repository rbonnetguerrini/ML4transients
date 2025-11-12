import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import os
from .models import CustomCNN
from .losses import get_loss_function
import torch.nn.functional as F
from pathlib import Path

class BaseTrainer(ABC):
    """Base trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup TensorBoard
        self.setup_tensorboard()
        self.setup_training()
        self._init_early_stopping()
    
    def setup_tensorboard(self):
        """Setup TensorBoard logging"""
        if self.config.get('use_tensorboard', True):
            log_dir = self.config.get('tensorboard_log_dir', 'runs')
            experiment_name = self.config.get('experiment_name', 'experiment')
            self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")
            print(f"TensorBoard logging to: {log_dir}/{experiment_name}")
            print(f"Run 'tensorboard --logdir={log_dir}' to view logs")
        else:
            self.writer = None
    
    def _init_early_stopping(self):
        es_cfg = self.config.get('early_stopping', {})
        self.es_enabled = es_cfg.get('enabled', False)
        self.es_monitor = es_cfg.get('monitor', 'loss')
        self.es_mode = es_cfg.get('mode', 'min')
        self.es_patience = es_cfg.get('patience', 100)
        self.es_min_delta = es_cfg.get('min_delta', 0.0)
        self.best_metric = float('-inf') if self.es_mode == 'max' else float('inf')
        self.epochs_no_improve = 0
        self.best_epoch = -1

    def _is_improvement(self, current):
        if current is None:
            return False
        if self.es_mode == 'max':
            return current > self.best_metric + self.es_min_delta
        else:
            return current < self.best_metric - self.es_min_delta

    def _get_current_lr(self):
        if hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        if hasattr(self, 'optimizers') and len(self.optimizers) > 0:
            return self.optimizers[0].param_groups[0]['lr']
        if hasattr(self, 'optimizer1'):
            return self.optimizer1.param_groups[0]['lr']
        return None

    @abstractmethod
    def setup_training(self):
        """Setup models, optimizers, loss functions"""
        pass
    
    @abstractmethod
    def train_one_epoch(self, epoch, train_loader):
        """Train for one epoch"""
        pass
    
    @abstractmethod
    def evaluate(self, test_loader):
        """Evaluate model"""
        pass
    
    def fit(self, train_loader, val_loader=None, test_loader=None):
        """Main training loop (order: train, val, test)."""
        for epoch in range(self.config['epochs']):
            train_metrics = self.train_one_epoch(epoch, train_loader)
            val_metrics = self.evaluate(val_loader) if val_loader else {}
            test_metrics = self.evaluate(test_loader) if test_loader else {}

            # Select metrics source for monitoring - prefer validation, fallback to test
            monitor_source = val_metrics if val_metrics else test_metrics
            current_monitored = monitor_source.get(self.es_monitor)

            # TensorBoard logging
            self.log_tensorboard(epoch, train_metrics, test_metrics, val_metrics)

            # Console logging
            self.log_epoch(epoch, train_metrics, test_metrics, val_metrics)

            # Early stopping / best checkpoint
            if current_monitored is not None:
                if self._is_improvement(current_monitored):
                    self.best_metric = current_monitored
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                    self.save_checkpoint(epoch, 'best')
                    print(f"New best {self.es_monitor}: {current_monitored:.6f} at epoch {epoch+1}")
                else:
                    self.epochs_no_improve += 1

                # Simple early stopping based on epochs without improvement
                if self.es_enabled and self.epochs_no_improve >= self.es_patience:
                    print(f"Early stopping: {self.epochs_no_improve} epochs without improvement")
                    print(f"Early stopping at epoch {epoch+1} (best {self.es_monitor}: {self.best_metric:.6f} at epoch {self.best_epoch+1})")
                    break

        # Save final model
        self.save_checkpoint(epoch, 'final')
        if self.writer:
            self.writer.close()
        return self.best_metric if self.best_epoch >= 0 else (current_monitored if current_monitored is not None else float('nan'))
    
    def log_tensorboard(self, epoch, train_metrics, test_metrics, val_metrics=None):
        """Log metrics to TensorBoard"""
        if not self.writer:
            return
            
        # Log training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Log test metrics
        for key, value in test_metrics.items():
            self.writer.add_scalar(f'Test/{key}', value, epoch)
        
        # Log validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
        
        # Log learning rate 
        current_lr = self._get_current_lr()
        if current_lr is not None:
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def log_epoch(self, epoch, train_metrics, test_metrics, val_metrics=None):
        """Log epoch results"""
        msg = f"Epoch {epoch+1}/{self.config['epochs']}"
        msg += f" - Train: {train_metrics}"
        msg += f" - Test: {test_metrics}"
        if val_metrics:
            msg += f" - Val: {val_metrics}"
        print(msg)
    
    @abstractmethod
    def save_checkpoint(self, epoch, suffix):
        """Save model checkpoint"""
        pass
    
    def accuracy(self, logits, targets):
        """Calculate accuracy"""
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct = (preds.squeeze() == targets).sum().item()
        return correct / len(targets)
    
    def compute_confusion_metrics(self, predictions, labels):
        """Compute confusion matrix metrics including FNR"""
        predictions = predictions.cpu() if torch.is_tensor(predictions) else predictions
        labels = labels.cpu() if torch.is_tensor(labels) else labels
        
        # Convert to binary predictions if not already
        if torch.is_tensor(predictions):
            preds = (predictions > 0.5).float()
        else:
            preds = (predictions > 0.5).astype(float)
        
        # Calculate confusion matrix components
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        return {
            'accuracy': accuracy,
            'fnr': fnr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }


class StandardTrainer(BaseTrainer):
    """Standard single model trainer"""
    
    def setup_training(self):
        # Model
        self.model = CustomCNN(**self.config['model_params']).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Loss function
        self.loss_fn = get_loss_function('standard')
        
        # Learning rate schedule
        self.setup_lr_schedule()
    
    def setup_lr_schedule(self):
        """Setup learning rate schedule"""
        self.alpha_plan = [self.config['learning_rate']] * self.config['epochs']
        epoch_decay_start = self.config.get('epoch_decay_start', 80)
        
        for i in range(epoch_decay_start, self.config['epochs']):
            self.alpha_plan[i] = (
                float(self.config['epochs'] - i) / 
                (self.config['epochs'] - epoch_decay_start) * 
                self.config['learning_rate']
            )

    def adjust_learning_rate(self, epoch):
        """Adjust learning rate according to schedule"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]

          
    def train_one_epoch(self, epoch, train_loader):
        self.model.train()
        self.adjust_learning_rate(epoch)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            if batch_idx >= self.config.get('num_iter_per_epoch', float('inf')):
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_correct += (torch.sigmoid(outputs.squeeze()) > 0.5).eq(labels).sum().item()
            total_samples += len(labels)
            
            # Log batch metrics to TensorBoard
            if self.writer and batch_idx % self.config.get('log_interval', 100) == 0:
                # Use fractional epoch for batch-level logging
                fractional_epoch = epoch + (batch_idx / len(train_loader))
                self.writer.add_scalar('Batch/Loss', loss.item(), fractional_epoch)
                batch_acc = (torch.sigmoid(outputs.squeeze()) > 0.5).eq(labels).sum().item() / len(labels)
                self.writer.add_scalar('Batch/Accuracy', batch_acc, fractional_epoch)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def evaluate(self, test_loader):
        if test_loader is None:
            return {}
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)
        return {
            'accuracy': total_correct / total_samples,
            'loss': total_loss / len(test_loader)
        }
    
    def save_checkpoint(self, epoch, suffix):
        torch.save(self.model.state_dict(), f"{self.config.get('output_dir')}/model_{suffix}.pth")


class CoTeachingTrainer(BaseTrainer):
    """Co-teaching trainer with two networks"""
    
    def setup_training(self):
        # Two models
        self.model1 = CustomCNN(**self.config['model_params']).to(self.device)
        self.model2 = CustomCNN(**self.config['model_params']).to(self.device)
        
        # Optimizers
        self.optimizer1 = torch.optim.Adam(
            self.model1.parameters(), 
            lr=self.config['learning_rate']
        )
        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Loss function
        self.loss_fn = get_loss_function(
            'coteaching',
            forget_rate_0=self.config.get('forget_rate_0', 0.015),
            forget_rate_1=self.config.get('forget_rate_1', 0.005)
        )
        
        # Learning rate and forget rate schedules
        self.setup_schedules()
    
    def setup_schedules(self):
        """Setup learning rate and forget rate schedules"""
        epochs = self.config['epochs']
        lr = self.config['learning_rate']
        epoch_decay_start = self.config.get('epoch_decay_start', 80)
        
        # Learning rate schedule
        self.alpha_plan = [lr] * epochs
        self.beta1_plan = [0.9] * epochs
        
        for i in range(epoch_decay_start, epochs):
            self.alpha_plan[i] = float(epochs - i) / (epochs - epoch_decay_start) * lr
            self.beta1_plan[i] = 0.1
        
        # Forget rate schedules
        num_gradual = self.config.get('num_gradual', 10)
        forget_rate_0 = self.config.get('forget_rate_0', 0.015)
        forget_rate_1 = self.config.get('forget_rate_1', 0.005)
        exponent = self.config.get('exponent', 1)
        
        self.rate_schedule_0 = np.ones(epochs) * forget_rate_0
        self.rate_schedule_0[:num_gradual] = np.linspace(0, forget_rate_0**exponent, num_gradual)
        
        self.rate_schedule_1 = np.ones(epochs) * forget_rate_1
        self.rate_schedule_1[:num_gradual] = np.linspace(0, forget_rate_1**exponent, num_gradual)
    
    def adjust_learning_rate(self, optimizer, epoch):
        """Adjust learning rate"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)
    
    def train_one_epoch(self, epoch, train_loader):
        self.model1.train()
        self.model2.train()
        self.adjust_learning_rate(self.optimizer1, epoch)
        self.adjust_learning_rate(self.optimizer2, epoch)
        
        total_correct1, total_correct2 = 0, 0
        total_samples = 0
        total_loss1, total_loss2 = 0.0, 0.0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            if batch_idx >= self.config.get('num_iter_per_epoch', float('inf')):
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            logits1 = self.model1(images)
            logits2 = self.model2(images)
            
            # Calculate accuracy
            total_correct1 += (torch.sigmoid(logits1.squeeze()) > 0.5).eq(labels).sum().item()
            total_correct2 += (torch.sigmoid(logits2.squeeze()) > 0.5).eq(labels).sum().item()
            total_samples += len(labels)
            
            # Co-teaching loss
            epoch_forget_rates = (self.rate_schedule_0[epoch], self.rate_schedule_1[epoch])
            loss_1, loss_2 = self.loss_fn(logits1, logits2, labels, epoch_forget_rates=epoch_forget_rates)
            
            total_loss1 += loss_1.item()
            total_loss2 += loss_2.item()
            
            # Backward pass
            self.optimizer1.zero_grad()
            loss_1.backward()
            self.optimizer1.step()
            
            self.optimizer2.zero_grad()
            loss_2.backward()
            self.optimizer2.step()
            
            # Log batch metrics to TensorBoard
            if self.writer and batch_idx % self.config.get('log_interval', 100) == 0:
                fractional_epoch = epoch + (batch_idx / len(train_loader))
                self.writer.add_scalar('Batch/Loss1', loss_1.item(), fractional_epoch)
                self.writer.add_scalar('Batch/Loss2', loss_2.item(), fractional_epoch)
        
        return {
            'accuracy1': total_correct1 / total_samples,
            'accuracy2': total_correct2 / total_samples,
            'loss1': total_loss1 / len(train_loader),
            'loss2': total_loss2 / len(train_loader)
        }
    
    def evaluate(self, test_loader):
        if test_loader is None:
            return {}
        self.model1.eval()
        self.model2.eval()
        
        all_preds1 = []
        all_preds2 = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits1 = self.model1(images)
                logits2 = self.model2(images)
                loss1 = F.binary_cross_entropy_with_logits(logits1.squeeze(), labels.float())
                loss2 = F.binary_cross_entropy_with_logits(logits2.squeeze(), labels.float())
                total_loss += 0.5 * (loss1.item() + loss2.item())
                
                preds1 = torch.sigmoid(logits1.squeeze())
                preds2 = torch.sigmoid(logits2.squeeze())
                all_preds1.append(preds1)
                all_preds2.append(preds2)
                all_labels.append(labels)
        
        # Concatenate all predictions and labels
        all_preds1 = torch.cat(all_preds1)
        all_preds2 = torch.cat(all_preds2)
        all_labels = torch.cat(all_labels)
        
        # Ensemble prediction (average of both models)
        ensemble_preds = (all_preds1 + all_preds2) / 2
        
        # Compute metrics for ensemble
        metrics = self.compute_confusion_metrics(ensemble_preds, all_labels)
        
        # Also compute individual model metrics
        metrics1 = self.compute_confusion_metrics(all_preds1, all_labels)
        metrics2 = self.compute_confusion_metrics(all_preds2, all_labels)
        
        return {
            'accuracy': metrics['accuracy'],
            'fnr': metrics['fnr'],
            'loss': total_loss / len(test_loader),
            'accuracy1': metrics1['accuracy'],
            'accuracy2': metrics2['accuracy'],
            'fnr1': metrics1['fnr'],
            'fnr2': metrics2['fnr']
        }
    
    def save_checkpoint(self, epoch, suffix):
        torch.save(self.model1.state_dict(), f"{self.config.get('output_dir')}/model1_{suffix}.pth")
        torch.save(self.model2.state_dict(), f"{self.config.get('output_dir')}/model2_{suffix}.pth")


class EnsembleTrainer(BaseTrainer):
    """Ensemble trainer with multiple models"""
    
    def setup_training(self):
        self.num_models = self.config.get('num_models', 3)
        
        # Multiple models
        self.models = []
        self.optimizers = []
        
        for i in range(self.num_models):
            model = CustomCNN(**self.config['model_params']).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            self.models.append(model)
            self.optimizers.append(optimizer)
        
        # Loss function
        self.loss_fn = get_loss_function('standard')
        
        # Learning rate schedule
        self.setup_lr_schedule()
    
    def setup_lr_schedule(self):
        """Setup learning rate schedule"""
        self.alpha_plan = [self.config['learning_rate']] * self.config['epochs']
        epoch_decay_start = self.config.get('epoch_decay_start', 80)
        
        for i in range(epoch_decay_start, self.config['epochs']):
            self.alpha_plan[i] = (
                float(self.config['epochs'] - i) / 
                (self.config['epochs'] - epoch_decay_start) * 
                self.config['learning_rate']
            )
    
    def adjust_learning_rate(self, epoch):
        """Adjust learning rate for all optimizers"""
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.alpha_plan[epoch]
    
    def train_one_epoch(self, epoch, train_loader):
        for model in self.models:
            model.train()
        self.adjust_learning_rate(epoch)
        
        model_accuracies = [0] * self.num_models
        model_losses = [0.0] * self.num_models
        total_samples = 0
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            if batch_idx >= self.config.get('num_iter_per_epoch', float('inf')):
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                model_accuracies[i] += (torch.sigmoid(outputs.squeeze()) > 0.5).eq(labels).sum().item()
                model_losses[i] += loss.item()
            
            total_samples += len(labels)
            # Average accuracy and loss across models
            if self.writer and batch_idx % self.config.get('log_interval', 100) == 0:
                fractional_epoch = epoch + (batch_idx / len(train_loader))
                avg_loss = sum(loss / (batch_idx + 1) for loss in model_losses) / self.num_models
                self.writer.add_scalar('Batch/Average_Loss', avg_loss, fractional_epoch)
        
        # Average accuracy and loss across models
        avg_accuracy = sum(acc / total_samples for acc in model_accuracies) / self.num_models
        avg_loss = sum(loss / len(train_loader) for loss in model_losses) / self.num_models
        return {
            'accuracy': avg_accuracy,
            'loss': avg_loss
        }
    
    def evaluate(self, test_loader):
        if test_loader is None:
            return {}
        for model in self.models:
            model.eval()
        
        all_ensemble_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions from all models
                batch_preds = []
                batch_losses = []
                for model in self.models:
                    outputs = model(images)
                    loss = self.loss_fn(outputs, labels)
                    batch_losses.append(loss.item())
                    preds = torch.sigmoid(outputs.squeeze())
                    batch_preds.append(preds)
                
                # Ensemble prediction (average of all models)
                ensemble_preds = torch.stack(batch_preds).mean(dim=0)
                all_ensemble_preds.append(ensemble_preds)
                all_labels.append(labels)
                total_loss += np.mean(batch_losses)
        
        # Concatenate all predictions and labels
        all_ensemble_preds = torch.cat(all_ensemble_preds)
        all_labels = torch.cat(all_labels)
        
        # Compute metrics
        metrics = self.compute_confusion_metrics(all_ensemble_preds, all_labels)
        metrics['loss'] = total_loss / len(test_loader)
        
        return metrics
    
    def save_checkpoint(self, epoch, suffix):
        output_dir = self.config.get('output_dir')
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{self.config.get('output_dir')}/ensemble_model_{i}_{suffix}.pth")


class StochasticCoTeachingTrainer(BaseTrainer):
    """Stochastic co-teaching trainer with two networks
    
    Based on: Jansen et al. (2023) - Stochastic co-teaching for training neural 
    networks with unknown levels of label noise.
    https://www.nature.com/articles/s41598-023-43864-7
    """
    
    def setup_training(self):
        # Two models
        self.model1 = CustomCNN(**self.config['model_params']).to(self.device)
        self.model2 = CustomCNN(**self.config['model_params']).to(self.device)
        
        # Optimizers
        self.optimizer1 = torch.optim.Adam(
            self.model1.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0)
        )
        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0)
        )
        
        # Loss function with stochastic co-teaching
        self.loss_fn = get_loss_function(
            'stochastic_coteaching',
            alpha=self.config.get('alpha', 32),
            beta=self.config.get('beta', 4),
            max_iters=self.config['epochs'],
            tp_gradual=self.config.get('num_gradual', 10),
            delay=self.config.get('delay', 0),
            exponent=self.config.get('exponent', 1),
            clip=self.config.get('clip', (0.01, 0.99)),
            seed=self.config.get('seed', 808)
        )
        
        # Learning rate schedule (optional)
        self.setup_lr_schedule()
    
    def setup_lr_schedule(self):
        """Setup learning rate schedule"""
        epochs = self.config['epochs']
        lr = self.config['learning_rate']
        epoch_decay_start = self.config.get('epoch_decay_start', 80)
        
        # Learning rate schedule
        self.alpha_plan = [lr] * epochs
        self.beta1_plan = [0.9] * epochs
        
        for i in range(epoch_decay_start, epochs):
            self.alpha_plan[i] = float(epochs - i) / (epochs - epoch_decay_start) * lr
            self.beta1_plan[i] = 0.1
    
    def adjust_learning_rate(self, optimizer, epoch):
        """Adjust learning rate"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)
    
    def train_one_epoch(self, epoch, train_loader):
        self.model1.train()
        self.model2.train()
        self.adjust_learning_rate(self.optimizer1, epoch)
        self.adjust_learning_rate(self.optimizer2, epoch)
        
        total_correct1, total_correct2 = 0, 0
        total_samples = 0
        total_loss1, total_loss2 = 0.0, 0.0
        total_reject1, total_reject2 = 0.0, 0.0
        batches_rejected = 0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            if batch_idx >= self.config.get('num_iter_per_epoch', float('inf')):
                break
                
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            logits1 = self.model1(images)
            logits2 = self.model2(images)
            
            # Calculate accuracy before loss
            total_correct1 += (torch.sigmoid(logits1.squeeze()) > 0.5).eq(labels).sum().item()
            total_correct2 += (torch.sigmoid(logits2.squeeze()) > 0.5).eq(labels).sum().item()
            total_samples += len(labels)
            
            # Stochastic co-teaching loss
            try:
                loss_1, loss_2 = self.loss_fn(logits1, logits2, labels)
            except RuntimeError as e:
                # Handle case where >90% samples are rejected
                batches_rejected += 1
                if batches_rejected > len(train_loader) * 0.1:  # If >10% batches rejected, raise error
                    raise RuntimeError(f"Too many batches rejected in epoch {epoch}: {batches_rejected}")
                continue
            
            total_loss1 += loss_1.item()
            total_loss2 += loss_2.item()
            
            # Track rejection rates
            frac_reject_1, frac_reject_2 = self.loss_fn.current_fraction_rejected()
            total_reject1 += frac_reject_1
            total_reject2 += frac_reject_2
            
            # Backward pass
            self.optimizer1.zero_grad()
            loss_1.backward()
            self.optimizer1.step()
            
            self.optimizer2.zero_grad()
            loss_2.backward()
            self.optimizer2.step()
            
            # Log batch metrics to TensorBoard
            if self.writer and batch_idx % self.config.get('log_interval', 100) == 0:
                fractional_epoch = epoch + (batch_idx / len(train_loader))
                self.writer.add_scalar('Batch/Loss1', loss_1.item(), fractional_epoch)
                self.writer.add_scalar('Batch/Loss2', loss_2.item(), fractional_epoch)
                self.writer.add_scalar('Batch/Reject1', frac_reject_1, fractional_epoch)
                self.writer.add_scalar('Batch/Reject2', frac_reject_2, fractional_epoch)
        
        # Step the loss function (increments internal iteration counter)
        self.loss_fn.step()
        
        # Calculate average metrics
        num_batches = min(len(train_loader), self.config.get('num_iter_per_epoch', float('inf'))) - batches_rejected
        
        metrics = {
            'accuracy1': total_correct1 / total_samples,
            'accuracy2': total_correct2 / total_samples,
            'loss1': total_loss1 / num_batches,
            'loss2': total_loss2 / num_batches,
            'reject_rate1': total_reject1 / num_batches,
            'reject_rate2': total_reject2 / num_batches,
        }
        
        if batches_rejected > 0:
            metrics['batches_rejected'] = batches_rejected
            print(f"Warning: {batches_rejected} batches rejected in epoch {epoch}")
        
        return metrics
    
    def evaluate(self, test_loader):
        if test_loader is None:
            return {}
        
        self.model1.eval()
        self.model2.eval()
        
        all_preds1 = []
        all_preds2 = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits1 = self.model1(images)
                logits2 = self.model2(images)
                
                # Standard loss for evaluation (no sample selection)
                loss1 = F.binary_cross_entropy_with_logits(logits1.squeeze(), labels.float())
                loss2 = F.binary_cross_entropy_with_logits(logits2.squeeze(), labels.float())
                total_loss += 0.5 * (loss1.item() + loss2.item())
                
                preds1 = torch.sigmoid(logits1.squeeze())
                preds2 = torch.sigmoid(logits2.squeeze())
                all_preds1.append(preds1)
                all_preds2.append(preds2)
                all_labels.append(labels)
        
        # Concatenate all predictions and labels
        all_preds1 = torch.cat(all_preds1)
        all_preds2 = torch.cat(all_preds2)
        all_labels = torch.cat(all_labels)
        
        # Ensemble prediction (average of both models)
        ensemble_preds = (all_preds1 + all_preds2) / 2
        
        # Compute metrics for ensemble
        metrics = self.compute_confusion_metrics(ensemble_preds, all_labels)
        
        # Also compute individual model metrics
        metrics1 = self.compute_confusion_metrics(all_preds1, all_labels)
        metrics2 = self.compute_confusion_metrics(all_preds2, all_labels)
        
        return {
            'accuracy': metrics['accuracy'],
            'fnr': metrics['fnr'],
            'loss': total_loss / len(test_loader),
            'accuracy1': metrics1['accuracy'],
            'accuracy2': metrics2['accuracy'],
            'fnr1': metrics1['fnr'],
            'fnr2': metrics2['fnr']
        }
    
    def log_tensorboard(self, epoch, train_metrics, test_metrics, val_metrics=None):
        """Extended TensorBoard logging with rejection rates"""
        # Call parent method
        super().log_tensorboard(epoch, train_metrics, test_metrics, val_metrics)
        
        # Log stochastic co-teaching specific metrics
        if self.writer:
            # Log rejection rates
            if 'reject_rate1' in train_metrics:
                self.writer.add_scalar('Train/Reject_Rate1', train_metrics['reject_rate1'], epoch)
                self.writer.add_scalar('Train/Reject_Rate2', train_metrics['reject_rate2'], epoch)
            
            # Log probabilities histograms
            if self.loss_fn.probas1 is not None and self.loss_fn.probas2 is not None:
                self.writer.add_histogram('Probabilities/Training_1', 
                                         self.loss_fn.probas1.cpu().numpy(), epoch)
                self.writer.add_histogram('Probabilities/Training_2', 
                                         self.loss_fn.probas2.cpu().numpy(), epoch)
    
    def save_checkpoint(self, epoch, suffix):
        output_dir = self.config.get('output_dir')
        torch.save(self.model1.state_dict(), f"{output_dir}/stocot_model1_{suffix}.pth")
        torch.save(self.model2.state_dict(), f"{output_dir}/stocot_model2_{suffix}.pth")


def get_trainer(trainer_type, config):
    """Factory function to get trainer"""
    if trainer_type == "standard":
        return StandardTrainer(config)
    elif trainer_type == "coteaching":
        return CoTeachingTrainer(config)
    elif trainer_type == "stochastic_coteaching":
        return StochasticCoTeachingTrainer(config)
    elif trainer_type == "ensemble":
        return EnsembleTrainer(config)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")