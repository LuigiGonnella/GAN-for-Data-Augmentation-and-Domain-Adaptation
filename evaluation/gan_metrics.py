import torch
import torch.nn as nn
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Normalize
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm


class GANMetrics:
    
    def __init__(self, output_dir, device='cpu'):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.metrics_history = {
            'epoch': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_real': [],
            'discriminator_fake': [],
            'fid_score': [],
            'is_score_mean': [],
            'is_score_std': []
        }
        
        self._inception_model = None
        
    def update(self, epoch, g_loss, d_loss, d_real=None, d_fake=None):
        
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['generator_loss'].append(float(g_loss))
        self.metrics_history['discriminator_loss'].append(float(d_loss))
        
        if d_real is not None:
            self.metrics_history['discriminator_real'].append(float(d_real))
        
        if d_fake is not None:
            self.metrics_history['discriminator_fake'].append(float(d_fake))
    
    def update_fid_is(self, fid_score, is_score_mean, is_score_std):
        
        self.metrics_history['fid_score'].append(float(fid_score))
        self.metrics_history['is_score_mean'].append(float(is_score_mean))
        self.metrics_history['is_score_std'].append(float(is_score_std))
    
    def save_metrics(self, n_iter):
        if n_iter != False:
            csv_path = self.output_dir / f'training_metrics_iter_{n_iter}.csv'
        else:
            csv_path = self.output_dir / 'training_metrics.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_history.keys())
            writer.writeheader()
            
            for i in range(len(self.metrics_history['epoch'])):
                row = {}
                for key in self.metrics_history.keys():
                    if i < len(self.metrics_history[key]):
                        row[key] = self.metrics_history[key][i]
                writer.writerow(row)
        
        print(f"Metrics saved to {csv_path}")
    
    def get_inception_model(self):
        if self._inception_model is None:
            self._inception_model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False
            ).to(self.device)
            self._inception_model.eval()
        return self._inception_model
    
    def compute_inception_features(self, images):
        import torch.nn.functional as F
        
        model = self.get_inception_model()
        
        # Resize to 299x299 if needed (Inception V3 requirement)
        if images.shape[-1] != 299 or images.shape[-2] != 299:
            images = F.interpolate(
                images, 
                size=(299, 299), 
                mode='bilinear', 
                align_corners=False
            )
        
        images = (images + 1) / 2
        
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        images = normalize(images)
        
        with torch.no_grad():
            features = model(images)
            
            if isinstance(features, tuple):
                features = features[0]
            
            features = features.cpu().numpy()
        
        return features
    
    def compute_fid(self, real_images, fake_images):
        real_features = self.compute_inception_features(real_images)
        fake_features = self.compute_inception_features(fake_images)
        
        real_mean = np.mean(real_features, axis=0)
        fake_mean = np.mean(fake_features, axis=0)
        
        real_cov = np.cov(real_features.T)
        fake_cov = np.cov(fake_features.T)
        
        mean_diff = real_mean - fake_mean
        
        try:
            cov_mean = self._sqrtm(fake_cov @ real_cov)
            if np.any(np.isnan(cov_mean)):
                cov_mean = np.zeros_like(fake_cov)
        except np.linalg.LinAlgError:
            return float('inf')
        
        fid = np.sum(mean_diff ** 2) + np.trace(real_cov + fake_cov - 2 * cov_mean)
        
        if np.isnan(fid) or np.isinf(fid):
            return float('inf')
        
        return float(fid)
    
    def compute_inception_score(self, fake_images, n_splits=10):
        features = self.compute_inception_features(fake_images)
        
        n_samples = features.shape[0]
        split_size = n_samples // n_splits
        
        scores = []
        for i in range(n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < n_splits - 1 else n_samples
            
            chunk_features = features[start:end]
            
            probs = np.exp(chunk_features) / np.sum(np.exp(chunk_features), axis=1, keepdims=True)
            
            py = np.mean(probs, axis=0)
            kl_divs = probs * (np.log(probs + 1e-10) - np.log(py + 1e-10))
            is_score = np.exp(np.mean(np.sum(kl_divs, axis=1)))
            
            scores.append(is_score)
        
        return np.mean(scores), np.std(scores)
    
    def detect_mode_collapse(self, fake_images, threshold=0.7):
        """
        Detect mode collapse by analyzing feature diversity using cosine similarity
        
        """
        features = self.compute_inception_features(fake_images)
        
        features_norm = np.linalg.norm(features, axis=1, keepdims=True)
        features_normalized = features / (features_norm + 1e-10)
        
        similarities = np.dot(features_normalized, features_normalized.T)
        
        mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
        mean_similarity = np.mean(similarities[mask])
        
        mean_similarity = np.clip(mean_similarity, -1.0, 1.0)
        
        diversity_score = 1 - mean_similarity
        
        return {
            'is_collapsed': mean_similarity > threshold,
            'diversity_score': float(diversity_score),
            'mean_similarity': float(mean_similarity),
            'threshold': threshold
        }
    
    def detect_vanishing_gradients(self, window_size=10):
        """
        Detect vanishing gradients by analyzing loss trajectory changes.
        Vanishing gradients = losses plateau (minimal change over epochs)
        
        """
        g_losses = np.array(self.metrics_history['generator_loss'])
        d_losses = np.array(self.metrics_history['discriminator_loss'])
        
        if len(g_losses) < window_size + 1:
            return {
                'has_vanishing_gradients': False,
                'g_gradient_magnitude': 0.0,
                'd_gradient_magnitude': 0.0,
                'recent_epochs': len(g_losses),
                'warning': 'Not enough epochs to detect'
            }
        
        recent_g_losses = g_losses[-window_size:]
        recent_d_losses = d_losses[-window_size:]
        
        g_gradients = np.abs(np.diff(recent_g_losses))
        d_gradients = np.abs(np.diff(recent_d_losses))
        
        g_gradient_magnitude = float(np.mean(g_gradients))
        d_gradient_magnitude = float(np.mean(d_gradients))
        
        vanishing_threshold = 0.001
        has_vanishing = (g_gradient_magnitude < vanishing_threshold) or (d_gradient_magnitude < vanishing_threshold)
        
        return {
            'has_vanishing_gradients': has_vanishing,
            'g_gradient_magnitude': g_gradient_magnitude,
            'd_gradient_magnitude': d_gradient_magnitude,
            'recent_epochs': window_size,
            'threshold': vanishing_threshold
        }
    
    def plot_losses(self, save_path=None, n_iter=False):
        """
        Plot generator and discriminator losses over time
        
        """
        if save_path is None:
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            if n_iter != False:
                save_path = plots_dir / f'losses_iter_{n_iter}.png'
            else:
                save_path = plots_dir / 'losses.png'
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        epochs = self.metrics_history['epoch']
        g_losses = self.metrics_history['generator_loss']
        d_losses = self.metrics_history['discriminator_loss']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, g_losses, label='Generator Loss', linewidth=2)
        plt.plot(epochs, d_losses, label='Discriminator Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Losses', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if self.metrics_history['discriminator_real'] and self.metrics_history['discriminator_fake']:
            d_real = self.metrics_history['discriminator_real']
            d_fake = self.metrics_history['discriminator_fake']
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs, d_real, label='D(Real)', linewidth=2)
            plt.plot(epochs, d_fake, label='D(Fake)', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Discriminator Output', fontsize=12)
            plt.title('Discriminator Confidence', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plots saved to {save_path}")
    
    def plot_quality_metrics(self, fid_scores=None, is_scores=None, save_path=None, n_iter=False):

        if save_path is None:
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            if n_iter != False:
                save_path = plots_dir / f'quality_metrics_iter_{n_iter}.png'
            else:
                save_path = plots_dir / 'quality_metrics.png'
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if fid_scores is not None and len(fid_scores) > 0:
            epochs = range(len(fid_scores))
            axes[0].plot(epochs, fid_scores, marker='o', linewidth=2, markersize=4)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('FID Score', fontsize=12)
            axes[0].set_title('Fréchet Inception Distance (lower is better)', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        
        if is_scores is not None and len(is_scores) > 0:
            epochs = range(len(is_scores))
            means = [s[0] for s in is_scores]
            stds = [s[1] for s in is_scores]
            
            axes[1].errorbar(epochs, means, yerr=stds, marker='o', linewidth=2, 
                           capsize=5, markersize=4, label='IS ± std')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Inception Score', fontsize=12)
            axes[1].set_title('Inception Score (higher is better)', fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quality metric plots saved to {save_path}")
    
    @staticmethod
    def _sqrtm(matrix):
        """Compute matrix square root using scipy"""
        return sqrtm(matrix).real


def get_discriminator_stats(d_real, d_fake):
    return {
        'real_mean': float(torch.mean(d_real).detach().cpu()),
        'real_std': float(torch.std(d_real).detach().cpu()),
        'fake_mean': float(torch.mean(d_fake).detach().cpu()),
        'fake_std': float(torch.std(d_fake).detach().cpu()),
    }
