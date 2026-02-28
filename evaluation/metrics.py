"""
evaluation/metrics.py
Evaluation metrics for RNFL thickness prediction
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List


def compute_rnfl_metrics(predictions: torch.Tensor, 
                         targets: torch.Tensor,
                         metadata: List[Dict] = None) -> Dict:
    """
    Compute comprehensive evaluation metrics for RNFL prediction.
    
    Metrics include:
    - Point-wise accuracy (MAE, RMSE)
    - Shape fidelity (Pearson R, R², CCC)
    - Gradient preservation (Gradient MAE)
    - Prediction variance (for template overfitting check)
    - Demographic fairness (if metadata provided)
    
    Args:
        predictions: Predicted RNFL profiles (B, 360) or numpy array
        targets: Ground truth RNFL profiles (B, 360) or numpy array
        metadata: Optional list of dicts with demographic info
                  Keys: 'age', 'gender', 'race'
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # 1. Point-wise Accuracy Metrics
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    
    metrics['MAE'] = mae
    metrics['RMSE'] = rmse
    
    # 2. Gradient (Shape/Slope) Preservation
    pred_grad = predictions - np.roll(predictions, shift=1, axis=1)
    target_grad = targets - np.roll(targets, shift=1, axis=1)
    grad_mae = float(np.mean(np.abs(pred_grad - target_grad)))
    
    metrics['Grad_MAE'] = grad_mae
    
    # 3. Shape and Scale Metrics (Per-sample, then averaged)
    pearson_rs = []
    r2_scores = []
    cccs = []
    
    for pred, tgt in zip(predictions, targets):
        # Check for sufficient variance
        if pred.std() > 1e-6 and tgt.std() > 1e-6:
            # Pearson Correlation
            r = np.corrcoef(pred, tgt)[0, 1]
            if not np.isnan(r):
                pearson_rs.append(r)
            
            # R² Score
            ss_res = ((tgt - pred) ** 2).sum()
            ss_tot = ((tgt - tgt.mean()) ** 2).sum()
            if ss_tot > 1e-6:
                r2 = 1 - (ss_res / ss_tot)
                r2_scores.append(r2)
            
            # Concordance Correlation Coefficient (CCC)
            mean_pred, mean_tgt = pred.mean(), tgt.mean()
            var_pred, var_tgt = pred.var(), tgt.var()
            covariance = np.cov(pred, tgt)[0, 1]
            
            denominator = var_pred + var_tgt + (mean_pred - mean_tgt) ** 2
            if denominator > 1e-6:
                ccc = (2 * covariance) / denominator
                if not np.isnan(ccc):
                    cccs.append(ccc)
    
    metrics['Pearson_R'] = float(np.mean(pearson_rs)) if pearson_rs else 0.0
    metrics['R2'] = float(np.mean(r2_scores)) if r2_scores else 0.0
    metrics['CCC'] = float(np.mean(cccs)) if cccs else 0.0
    
    # 4. Prediction Variance (Template Overfitting Check)
    # Global variance across all predictions
    sigma_pred = float(np.std(predictions))
    sigma_target = float(np.std(targets))
    
    metrics['sigma_pred'] = sigma_pred
    metrics['sigma_target'] = sigma_target
    metrics['variance_ratio'] = sigma_pred / sigma_target if sigma_target > 0 else 0.0
    
    # 5. Demographic Fairness (if metadata provided)
    if metadata is not None:
        fairness_metrics = compute_fairness_metrics(predictions, targets, metadata)
        metrics.update(fairness_metrics)
    
    return metrics


def compute_fairness_metrics(predictions: np.ndarray,
                             targets: np.ndarray, 
                             metadata: List[Dict]) -> Dict:
    """
    Compute demographic fairness metrics.
    
    Measures performance gaps across:
    - Age groups: <40, 40-60, >60
    - Gender: Male, Female
    - Race: Black, Asian, Hispanic, White
    
    Args:
        predictions: Predicted RNFL profiles (B, 360)
        targets: Ground truth RNFL profiles (B, 360)
        metadata: List of dicts with 'age', 'gender', 'race'
    
    Returns:
        Dictionary with fairness metrics
    """
    # Per-sample MAE
    per_sample_mae = np.mean(np.abs(predictions - targets), axis=1)
    
    # Extract demographic attributes
    ages = np.array([m.get('age', 50) for m in metadata])
    genders = np.array([m.get('gender', 'U') for m in metadata])
    races = np.array([m.get('race', 0) for m in metadata])
    
    fairness_metrics = {}
    group_maes = {}
    
    # Helper function to compute gap for a demographic attribute
    def compute_gap(masks, labels):
        """
        Compute max - min MAE across groups.
        """
        group_means = []
        for mask, label in zip(masks, labels):
            if mask.sum() >= 10:  # At least 10 samples
                mae_val = float(per_sample_mae[mask].mean())
                group_maes[label] = mae_val
                group_means.append(mae_val)
        
        if len(group_means) >= 2:
            return max(group_means) - min(group_means)
        else:
            return 0.0
    
    # Age gap
    age_masks = [ages < 40, (ages >= 40) & (ages < 60), ages >= 60]
    age_labels = ['age<40', 'age40-60', 'age>60']
    age_gap = compute_gap(age_masks, age_labels)
    
    # Gender gap
    gender_masks = [genders == 'M', genders == 'F']
    gender_labels = ['male', 'female']
    gender_gap = compute_gap(gender_masks, gender_labels)
    
    # Race gap
    race_masks = [races == 1, races == 2, races == 3, races == 4]
    race_labels = ['black', 'asian', 'hispanic', 'white']
    race_gap = compute_gap(race_masks, race_labels)
    
    # Overall fairness gap (average of individual gaps)
    gaps = [g for g in [age_gap, gender_gap, race_gap] if g > 0]
    avg_gap = float(np.mean(gaps)) if gaps else 0.0
    
    fairness_metrics['age_gap'] = age_gap
    fairness_metrics['gender_gap'] = gender_gap
    fairness_metrics['race_gap'] = race_gap
    fairness_metrics['avg_fairness_gap'] = avg_gap
    fairness_metrics['group_maes'] = group_maes
    
    return fairness_metrics


def compute_sector_metrics(predictions: np.ndarray,
                           targets: np.ndarray) -> Dict:
    """
    Compute metrics for RNFL sectors (Superior, Nasal, Inferior, Temporal).
    
    Standard sector definitions (degrees):
    - Superior: 45-135
    - Nasal: 135-225
    - Inferior: 225-315
    - Temporal: 315-45 (wraps around)
    
    Args:
        predictions: Predicted RNFL profiles (B, 360)
        targets: Ground truth RNFL profiles (B, 360)
    
    Returns:
        Dictionary with sector-wise MAE
    """
    sector_metrics = {}
    
    # Define sector indices
    sectors = {
        'Superior': np.arange(45, 135),
        'Nasal': np.arange(135, 225),
        'Inferior': np.arange(225, 315),
        'Temporal': np.concatenate([np.arange(315, 360), np.arange(0, 45)])
    }
    
    # Compute MAE for each sector
    for sector_name, indices in sectors.items():
        sector_pred = predictions[:, indices]
        sector_target = targets[:, indices]
        sector_mae = float(np.mean(np.abs(sector_pred - sector_target)))
        sector_metrics[f'{sector_name}_MAE'] = sector_mae
    
    # Global MAE
    sector_metrics['Global_MAE'] = float(np.mean(np.abs(predictions - targets)))
    
    return sector_metrics


def check_template_overfitting(predictions: np.ndarray,
                               threshold_low=6.0,
                               threshold_high=10.0) -> Dict:
    """
    Check for template overfitting based on prediction variance.
    
    Template overfitting occurs when all predictions are nearly identical,
    indicating the model outputs a "safe" population mean.
    
    Thresholds:
    - σ < 6 μm: Severe overfitting (collapsed)
    - 6 < σ < 10: Moderate overfitting
    - σ > 10 μm: Healthy diversity
    
    Args:
        predictions: Predicted RNFL profiles (B, 360)
        threshold_low: Lower threshold for variance (default: 6.0)
        threshold_high: Upper threshold for variance (default: 10.0)
    
    Returns:
        Dictionary with overfitting status and metrics
    """
    sigma = np.std(predictions)
    
    if sigma < threshold_low:
        status = "SEVERE_OVERFITTING"
        message = f"Variance = {sigma:.1f} μm < {threshold_low} (COLLAPSED)"
    elif sigma < threshold_high:
        status = "MODERATE_OVERFITTING"
        message = f"Variance = {sigma:.1f} μm (Borderline)"
    else:
        status = "HEALTHY"
        message = f"Variance = {sigma:.1f} μm (Diversity preserved)"
    
    return {
        'status': status,
        'sigma_pred': float(sigma),
        'message': message,
        'is_overfitting': sigma < threshold_low
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing RNFL metrics...")
    print("=" * 50)
    
    # Generate synthetic data
    batch_size = 100
    n_points = 360
    
    predictions = np.random.randn(batch_size, n_points) * 10 + 100
    targets = np.random.randn(batch_size, n_points) * 10 + 100
    
    # Generate synthetic metadata
    metadata = []
    for i in range(batch_size):
        metadata.append({
            'age': np.random.randint(20, 80),
            'gender': np.random.choice(['M', 'F']),
            'race': np.random.choice([1, 2, 3, 4])
        })
    
    # Test main metrics
    metrics = compute_rnfl_metrics(predictions, targets, metadata)
    
    print("\nMain Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f} μm")
    print(f"  Gradient MAE: {metrics['Grad_MAE']:.2f}")
    print(f"  Pearson R: {metrics['Pearson_R']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  CCC: {metrics['CCC']:.4f}")
    
    print("\nVariance Metrics:")
    print(f"  σ(pred): {metrics['sigma_pred']:.2f} μm")
    print(f"  σ(target): {metrics['sigma_target']:.2f} μm")
    print(f"  Ratio: {metrics['variance_ratio']:.3f}")
    
    print("\nFairness Metrics:")
    print(f"  Age gap: {metrics['age_gap']:.2f}")
    print(f"  Gender gap: {metrics['gender_gap']:.2f}")
    print(f"  Race gap: {metrics['race_gap']:.2f}")
    print(f"  Avg fairness gap: {metrics['avg_fairness_gap']:.2f}")
    
    # Test sector metrics
    sector_metrics = compute_sector_metrics(predictions, targets)
    print("\nSector Metrics:")
    for sector, mae in sector_metrics.items():
        print(f"  {sector}: {mae:.2f} μm")
    
    # Test template overfitting check
    overfitting_check = check_template_overfitting(predictions)
    print("\nTemplate Overfitting Check:")
    print(f"  Status: {overfitting_check['status']}")
    print(f"  {overfitting_check['message']}")
    
    print("\n✓ All metrics working!")
