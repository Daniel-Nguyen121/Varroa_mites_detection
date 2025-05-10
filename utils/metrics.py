"""
Metrics tracking and plotting utilities for training and validation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.general import Averager
import os
from typing import List, Optional, Union, Dict

class MetricsTracker:
    def __init__(self, out_dir: str):
        """Initialize metrics tracker with empty lists for all metrics"""
        self.train_loss_hist = Averager()
        self.val_loss_hist = Averager()
        self.train_loss_list: List[float] = []  # Per iteration
        self.train_loss_epoch: List[float] = []  # Per epoch
        self.val_loss_epoch: List[float] = []  # Per epoch
        self.val_map_50: List[float] = []
        self.val_map: List[float] = []
        
        # Component losses
        self.loss_cls_list: List[float] = []
        self.loss_box_reg_list: List[float] = []
        self.loss_objectness_list: List[float] = []
        self.loss_rpn_list: List[float] = []
        
        self.out_dir = out_dir
        self.best_map = 0.0
        
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

    def update(self, train_loss: Optional[float], val_metrics: Optional[tuple], epoch: int, 
              component_losses: Optional[tuple] = None):
        """Update metrics after each epoch"""
        # Get training loss from Averager
        epoch_loss = self.train_loss_hist.value
        self.train_loss_epoch.append(epoch_loss)
        
        # Update training metrics if available
        if train_loss is not None:
            self.train_loss_list.append(float(train_loss))
        
        # Update component losses if available
        if component_losses is not None:
            loss_cls, loss_box_reg, loss_objectness, loss_rpn = component_losses
            self.loss_cls_list.append(float(loss_cls))
            self.loss_box_reg_list.append(float(loss_box_reg))
            self.loss_objectness_list.append(float(loss_objectness))
            self.loss_rpn_list.append(float(loss_rpn))
        
        # Update validation metrics
        if val_metrics is not None:
            val_map_epoch, val_map_50_epoch, val_loss = val_metrics
            self.val_map.append(float(val_map_epoch))
            self.val_map_50.append(float(val_map_50_epoch))
            self.val_loss_epoch.append(float(val_loss))
            
            # Check if this is the best model
            if val_map_epoch > self.best_map:
                self.best_map = val_map_epoch
                print(f"\nBEST VALIDATION mAP: {self.best_map:.4f}")
                print(f"SAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
                return True
        return False

    def plot_metrics(self):
        """Plot and save all metrics"""
        # Plot training loss (per iteration)
        if self.train_loss_list:
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(len(self.train_loss_list)), self.train_loss_list, label='Training Loss')
            plt.title('Training Loss per Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, 'train_loss_iter.png'))
            plt.close()

        # Plot training vs validation metrics
        if self.train_loss_epoch:
            epochs = np.arange(1, len(self.train_loss_epoch) + 1)
            
            # Loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.train_loss_epoch, label='Training Loss')
            if self.val_loss_epoch:
                plt.plot(epochs, self.val_loss_epoch, label='Validation Loss')
            plt.title('Training and Validation Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, 'train_loss_epoch.png'))
            plt.close()

            # Component losses plot
            plt.figure(figsize=(10, 6))
            if self.loss_cls_list:
                plt.plot(epochs, self.loss_cls_list, label='Classification Loss')
            if self.loss_box_reg_list:
                plt.plot(epochs, self.loss_box_reg_list, label='Box Regression Loss')
            if self.loss_objectness_list:
                plt.plot(epochs, self.loss_objectness_list, label='Objectness Loss')
            if self.loss_rpn_list:
                plt.plot(epochs, self.loss_rpn_list, label='RPN Loss')
            plt.title('Component Losses per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, 'component_losses.png'))
            plt.close()

            # mAP plot
            if len(self.val_map) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, self.val_map_50, label='mAP@0.5')
                plt.plot(epochs, self.val_map, label='mAP@0.5:0.95')
                plt.title('Validation mAP')
                plt.xlabel('Epoch')
                plt.ylabel('mAP')
                plt.legend()
                plt.savefig(os.path.join(self.out_dir, 'map.png'))
                plt.close()

    def _plot_component_loss(self, filename, loss_values, title):
        """Helper to plot individual loss components"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.out_dir, filename))
        plt.close()

    def save_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                  epoch: int, out_dir: str, data_configs: Dict, model_name: str, is_best: bool = False):
        """Save model checkpoint with all metrics"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_list': self.train_loss_list,
            'train_loss_epoch': self.train_loss_epoch,
            'val_loss_epoch': self.val_loss_epoch,
            'val_map': self.val_map,
            'val_map_50': self.val_map_50,
            'loss_cls_list': self.loss_cls_list,
            'loss_box_reg_list': self.loss_box_reg_list,
            'loss_objectness_list': self.loss_objectness_list,
            'loss_rpn_list': self.loss_rpn_list,
            'data_configs': data_configs,
            'model_name': model_name
        }
        
        # Save last state
        torch.save(checkpoint, os.path.join(out_dir, 'last_model_state.pth'))
        
        # Save best model if current mAP is highest
        if is_best:
            torch.save(checkpoint, os.path.join(out_dir, 'best_model.pth'))

    def load_state(self, checkpoint: Dict):
        """Load metrics state from checkpoint"""
        self.train_loss_list = checkpoint.get('train_loss_list', [])
        self.train_loss_epoch = checkpoint.get('train_loss_epoch', [])
        self.val_loss_epoch = checkpoint.get('val_loss_epoch', [])
        self.val_map = checkpoint.get('val_map', [])
        self.val_map_50 = checkpoint.get('val_map_50', [])
        self.loss_cls_list = checkpoint.get('loss_cls_list', [])
        self.loss_box_reg_list = checkpoint.get('loss_box_reg_list', [])
        self.loss_objectness_list = checkpoint.get('loss_objectness_list', [])
        self.loss_rpn_list = checkpoint.get('loss_rpn_list', [])
        self.best_map = max(self.val_map) if self.val_map else 0.0 