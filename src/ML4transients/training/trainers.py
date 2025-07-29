import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import os
from .models import CustomCNN
from .losses import get_loss_function

class BaseTrainer(ABC):
    """Base trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup TensorBoard
        self.setup_tensorboard()
        self.setup_training()
    
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
    
    def fit(self, train_loader, test_loader, val_loader=None):
        """Main training loop"""
        best_acc = 0.0
        
        for epoch in range(self.config['epochs']):
            # Training
            train_metrics = self.train_one_epoch(epoch, train_loader)
            
            # Evaluation
            test_metrics = self.evaluate(test_loader)
            
            # Validation if available
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            
            # TensorBoard logging
            self.log_tensorboard(epoch, train_metrics, test_metrics, val_metrics)
            
            # Console logging
            self.log_epoch(epoch, train_metrics, test_metrics, val_metrics)
            
            # Save best model
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                self.save_checkpoint(epoch, 'best')
        
        # Save final model and close TensorBoard
        self.save_checkpoint(epoch, 'final')
        if self.writer:
            self.writer.close()
        return best_acc
    
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
        if hasattr(self, 'alpha_plan'):
            self.writer.add_scalar('Learning_Rate', self.alpha_plan[epoch], epoch)
    
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
        """Adjust learning rate"""
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
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Batch/Loss', loss.item(), step)
                batch_acc = (torch.sigmoid(outputs.squeeze()) > 0.5).eq(labels).sum().item() / len(labels)
                self.writer.add_scalar('Batch/Accuracy', batch_acc, step)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)
        
        return {'accuracy': total_correct / total_samples}
    
    def save_checkpoint(self, epoch, suffix):
        torch.save(self.model.state_dict(), f"model_{suffix}.pth")


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
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Batch/Loss1', loss_1.item(), step)
                self.writer.add_scalar('Batch/Loss2', loss_2.item(), step)
        
        return {
            'accuracy1': total_correct1 / total_samples,
            'accuracy2': total_correct2 / total_samples,
            'loss1': total_loss1 / len(train_loader),
            'loss2': total_loss2 / len(train_loader)
        }
    
    def evaluate(self, test_loader):
        self.model1.eval()
        self.model2.eval()
        
        total_correct1, total_correct2 = 0, 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits1 = self.model1(images)
                logits2 = self.model2(images)
                
                preds1 = (torch.sigmoid(logits1.squeeze()) > 0.5).float()
                preds2 = (torch.sigmoid(logits2.squeeze()) > 0.5).float()
                
                total_correct1 += (preds1 == labels).sum().item()
                total_correct2 += (preds2 == labels).sum().item()
                total_samples += len(labels)
        
        # Return ensemble accuracy (average of both models)
        ensemble_acc = (total_correct1 + total_correct2) / (2 * total_samples)
        return {
            'accuracy': ensemble_acc,
            'accuracy1': total_correct1 / total_samples,
            'accuracy2': total_correct2 / total_samples
        }
    
    def save_checkpoint(self, epoch, suffix):
        torch.save(self.model1.state_dict(), f"model1_{suffix}.pth")
        torch.save(self.model2.state_dict(), f"model2_{suffix}.pth")


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
                
                # Track accuracy and loss
                model_accuracies[i] += (torch.sigmoid(outputs.squeeze()) > 0.5).eq(labels).sum().item()
                model_losses[i] += loss.item()
            
            total_samples += len(labels)
            
            # Log batch metrics to TensorBoard
            if self.writer and batch_idx % self.config.get('log_interval', 100) == 0:
                step = epoch * len(train_loader) + batch_idx
                avg_loss = sum(model_losses) / (self.num_models * (batch_idx + 1))
                self.writer.add_scalar('Batch/Average_Loss', avg_loss, step)
        
        # Average accuracy and loss across models
        avg_accuracy = sum(acc / total_samples for acc in model_accuracies) / self.num_models
        avg_loss = sum(loss / len(train_loader) for loss in model_losses) / self.num_models
        
        return {
            'accuracy': avg_accuracy,
            'loss': avg_loss
        }
    
    def evaluate(self, test_loader):
        for model in self.models:
            model.eval()
        
        ensemble_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions from all models
                predictions = []
                for model in self.models:
                    outputs = model(images)
                    preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                    predictions.append(preds)
                
                # Ensemble prediction (majority vote)
                ensemble_preds = torch.stack(predictions).mean(dim=0)
                ensemble_preds = (ensemble_preds > 0.5).float()
                
                ensemble_correct += (ensemble_preds == labels).sum().item()
                total_samples += len(labels)
        
        return {'accuracy': ensemble_correct / total_samples}
    
    def save_checkpoint(self, epoch, suffix):
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"ensemble_model_{i}_{suffix}.pth")


def get_trainer(trainer_type, config):
    """Factory function to get trainer"""
    if trainer_type == "standard":
        return StandardTrainer(config)
    elif trainer_type == "coteaching":
        return CoTeachingTrainer(config)
    elif trainer_type == "ensemble":
        return EnsembleTrainer(config)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")