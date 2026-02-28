"""
losses/rnfl_losses.py
Loss functions for RNFL thickness prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAELoss(nn.Module):
    """
    Simple Mean Absolute Error loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted RNFL profiles (B, 360)
            targets: Ground truth RNFL profiles (B, 360)
        
        Returns:
            MAE loss (scalar)
        """
        return F.l1_loss(predictions, targets)


class GradientLoss(nn.Module):
    """
    Gradient-preserving loss for RNFL prediction.
    
    Penalizes deviations in spatial derivatives (slopes) rather than just
    point-wise errors. This preserves anatomical transitions (e.g., 
    glaucomatous notches) and prevents template overfitting.
    
    Loss = MAE + λ_g * ||∇T_pred - ∇T_gt||_1
    
    Args:
        lambda_gradient: Weight for gradient penalty (default: 0.2)
    """
    def __init__(self, lambda_gradient=0.2):
        super().__init__()
        self.lambda_gradient = lambda_gradient

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted RNFL profiles (B, 360)
            targets: Ground truth RNFL profiles (B, 360)
        
        Returns:
            Total loss (MAE + gradient penalty)
        """
        # Point-wise MAE
        mae_loss = F.l1_loss(predictions, targets)
        
        # Compute gradients (first-order differences with circular wrapping)
        pred_gradient = predictions - torch.roll(predictions, shifts=1, dims=1)
        target_gradient = targets - torch.roll(targets, shifts=1, dims=1)
        
        # Gradient MAE
        gradient_loss = F.l1_loss(pred_gradient, target_gradient)
        
        # Combined loss
        total_loss = mae_loss + self.lambda_gradient * gradient_loss
        
        return total_loss

    def get_components(self, predictions, targets):
        """
        Return individual loss components for logging.
        
        Returns:
            Dictionary with 'mae', 'gradient', and 'total' losses
        """
        mae_loss = F.l1_loss(predictions, targets)
        
        pred_gradient = predictions - torch.roll(predictions, shifts=1, dims=1)
        target_gradient = targets - torch.roll(targets, shifts=1, dims=1)
        gradient_loss = F.l1_loss(pred_gradient, target_gradient)
        
        total_loss = mae_loss + self.lambda_gradient * gradient_loss
        
        return {
            'mae': mae_loss.item(),
            'gradient': gradient_loss.item(),
            'total': total_loss.item()
        }


class PearsonLoss(nn.Module):
    """
    Pearson correlation loss (shape-preserving).
    
    WARNING: This loss can cause template overfitting by encouraging
    models to output smooth, canonical patterns.
    
    Loss = MAE + λ_p * (1 - Pearson(pred, target))
    
    Args:
        lambda_pearson: Weight for Pearson penalty (default: 0.5)
    """
    def __init__(self, lambda_pearson=0.5):
        super().__init__()
        self.lambda_pearson = lambda_pearson

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted RNFL profiles (B, 360)
            targets: Ground truth RNFL profiles (B, 360)
        
        Returns:
            Total loss (MAE + Pearson penalty)
        """
        # Point-wise MAE
        mae_loss = F.l1_loss(predictions, targets)
        
        # Pearson correlation
        pred_mean = predictions.mean(dim=1, keepdim=True)
        target_mean = targets.mean(dim=1, keepdim=True)
        
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt(
            (pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1)
        )
        
        # Avoid division by zero
        pearson = numerator / (denominator + 1e-8)
        pearson_loss = (1 - pearson).mean()
        
        # Combined loss
        total_loss = mae_loss + self.lambda_pearson * pearson_loss
        
        return total_loss


class StructuredLoss(nn.Module):
    """
    "Structured" protocol from paper (causes template overfitting).
    
    Combines MAE with Pearson correlation and aggressive Pearson weight.
    Used as a baseline to demonstrate template overfitting.
    
    Loss = MAE + λ_p * (1 - Pearson(pred, target))
    where λ_p ∈ {10, 30} causes severe variance collapse
    
    Args:
        lambda_pearson: Weight for Pearson penalty (default: 10.0)
    """
    def __init__(self, lambda_pearson=10.0):
        super().__init__()
        self.pearson_loss = PearsonLoss(lambda_pearson=lambda_pearson)

    def forward(self, predictions, targets):
        return self.pearson_loss(predictions, targets)


def get_loss_function(loss_type='gradient', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: Type of loss ('mae', 'gradient', 'pearson', 'structured')
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    loss_functions = {
        'mae': MAELoss,
        'gradient': GradientLoss,
        'pearson': PearsonLoss,
        'structured': StructuredLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)


if __name__ == "__main__":
    # Test loss functions
    batch_size = 4
    n_points = 360
    
    predictions = torch.randn(batch_size, n_points)
    targets = torch.randn(batch_size, n_points)
    
    print("Testing loss functions:")
    print("=" * 50)
    
    # Test MAE
    mae_loss = MAELoss()
    loss_mae = mae_loss(predictions, targets)
    print(f"MAE Loss: {loss_mae.item():.4f}")
    
    # Test Gradient Loss
    gradient_loss = GradientLoss(lambda_gradient=0.2)
    loss_grad = gradient_loss(predictions, targets)
    components = gradient_loss.get_components(predictions, targets)
    print(f"\nGradient Loss:")
    print(f"  MAE: {components['mae']:.4f}")
    print(f"  Gradient: {components['gradient']:.4f}")
    print(f"  Total: {components['total']:.4f}")
    
    # Test Pearson Loss
    pearson_loss = PearsonLoss(lambda_pearson=0.5)
    loss_pearson = pearson_loss(predictions, targets)
    print(f"\nPearson Loss: {loss_pearson.item():.4f}")
    
    # Test Structured Loss
    structured_loss = StructuredLoss(lambda_pearson=10.0)
    loss_structured = structured_loss(predictions, targets)
    print(f"\nStructured Loss: {loss_structured.item():.4f}")
    
    print("\n✓ All loss functions working!")
