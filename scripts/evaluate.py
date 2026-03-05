#!/usr/bin/env python3
"""
Verify Results Script - MICCAI 2026 Submission #4917
Reproduces paper metrics from pre-computed predictions

This script allows reviewers to verify paper results WITHOUT needing:
- Access to private datasets (FairFedMed, GRAPE)
- Trained model checkpoints (too large for GitHub)
- GPU resources for inference

Usage:
    # Verify main results (Table 1)
    python scripts/verify_results.py \
        --predictions results/fairfedmed_test_predictions.csv \
        --demographics results/fairfedmed_test_demographics.csv
    
    # Verify cross-modality results (Table 2)
    python scripts/verify_results.py \
        --predictions results/grape_predictions.csv
    
    # Verify all results at once
    python scripts/verify_results.py --verify_all
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_predictions(csv_path):
    """Load predictions from CSV file"""
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Verify required columns
    required_cols = ['sample_id', 'angle', 'prediction', 'ground_truth']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Reshape to (N_samples, 360)
    n_samples = df['sample_id'].nunique()
    n_angles = 360
    
    predictions = df.pivot(index='sample_id', columns='angle', values='prediction').values
    ground_truth = df.pivot(index='sample_id', columns='angle', values='ground_truth').values
    
    print(f"✓ Loaded {n_samples} samples × {n_angles} angles")
    return predictions, ground_truth


def load_demographics(csv_path):
    """Load demographic information if available"""
    if csv_path is None or not Path(csv_path).exists():
        return None
    
    print(f"Loading demographics from: {csv_path}")
    demo_df = pd.read_csv(csv_path)
    print(f"✓ Loaded demographics for {len(demo_df)} samples")
    return demo_df


def compute_metrics(predictions, ground_truth):
    """
    Compute all metrics reported in paper
    
    Returns dict with:
    - MAE (μm)
    - σ(pred), σ(GT) - Variance metrics
    - Pearson R
    - Sector-wise MAE
    """
    # Flatten for global metrics
    pred_flat = predictions.flatten()
    gt_flat = ground_truth.flatten()
    
    # Main metrics
    mae = np.mean(np.abs(pred_flat - gt_flat))
    
    # Variance metrics (KEY FOR TEMPLATE OVERFITTING!)
    pred_std = np.std(pred_flat)
    gt_std = np.std(gt_flat)
    variance_ratio = pred_std / gt_std if gt_std > 0 else 0.0
    
    # Pearson correlation
    pearson_r, p_value = stats.pearsonr(pred_flat, gt_flat)
    
    # Per-angle statistics
    per_angle_mae = np.mean(np.abs(predictions - ground_truth), axis=0)
    
    # Sector-wise analysis (standard RNFL sectors)
    def get_sector_mae(start, end):
        """Extract sector and compute MAE"""
        if end > start:
            sector_pred = predictions[:, start:end]
            sector_gt = ground_truth[:, start:end]
        else:  # Wraps around (temporal sector)
            sector_pred = np.concatenate([predictions[:, start:], predictions[:, :end]], axis=1)
            sector_gt = np.concatenate([ground_truth[:, start:], ground_truth[:, :end]], axis=1)
        return np.mean(np.abs(sector_pred - sector_gt))
    
    # Standard RNFL sectors
    superior_mae = get_sector_mae(46, 135)
    nasal_mae = get_sector_mae(136, 225)
    inferior_mae = get_sector_mae(226, 315)
    temporal_mae = get_sector_mae(316, 45)
    
    return {
        'mae': mae,
        'pred_std': pred_std,
        'gt_std': gt_std,
        'variance_ratio': variance_ratio,
        'pearson_r': pearson_r,
        'pearson_p': p_value,
        'per_angle_mae': per_angle_mae,
        'sectors': {
            'superior': superior_mae,
            'nasal': nasal_mae,
            'inferior': inferior_mae,
            'temporal': temporal_mae,
            'global': mae
        }
    }


def compute_fairness_metrics(predictions, ground_truth, demographics):
    """Compute demographic fairness gaps"""
    if demographics is None:
        return None
    
    fairness_results = {}
    
    # Flatten predictions for grouping
    pred_flat = predictions.flatten()
    gt_flat = ground_truth.flatten()
    
    # Repeat demographics for each angle (360 per sample)
    demo_repeated = pd.concat([demographics] * 360, ignore_index=True)
    
    for demo_col in ['race', 'gender', 'ethnicity']:
        if demo_col not in demographics.columns:
            continue
        
        group_maes = {}
        for group_value in demographics[demo_col].unique():
            # Get mask for this demographic group
            mask = demo_repeated[demo_col] == group_value
            
            # Compute MAE for this group
            group_mae = np.mean(np.abs(pred_flat[mask] - gt_flat[mask]))
            group_maes[str(group_value)] = group_mae
        
        # Fairness gap = max - min
        fairness_gap = max(group_maes.values()) - min(group_maes.values())
        
        fairness_results[demo_col] = {
            'group_maes': group_maes,
            'fairness_gap': fairness_gap
        }
    
    return fairness_results


def print_table_1_format(metrics, fairness=None, model_name="Gradient-Loss"):
    """Print results in Table 1 format from paper"""
    print("\n" + "="*70)
    print(f"TABLE 1 FORMAT - {model_name}")
    print("="*70)
    
    print("\nMetrics:")
    print(f"  MAE (μm):           {metrics['mae']:6.2f}")
    print(f"  Pearson R:          {metrics['pearson_r']:6.3f}")
    
    if fairness and 'race' in fairness:
        print(f"  Fairness Gap (μm):  {fairness['race']['fairness_gap']:6.2f}")
    
    print(f"  σ(pred) (μm):       {metrics['pred_std']:6.2f}")
    
    # Template overfitting check
    if metrics['pred_std'] < 6.0:
        print(f"  ⚠️  WARNING: Template overfitting detected! (σ < 6 μm)")
    else:
        print(f"  ✓  No template overfitting (σ = {metrics['pred_std']:.2f} μm)")
    
    print("="*70)


def print_table_2_format(metrics):
    """Print results in Table 2 format from paper (sector-wise)"""
    print("\n" + "="*70)
    print("TABLE 2 FORMAT - Sector-wise RNFL Regression")
    print("="*70)
    
    print("\nSector-wise MAE (μm):")
    print(f"  Global:     {metrics['sectors']['global']:6.2f}")
    print(f"  Superior:   {metrics['sectors']['superior']:6.2f}")
    print(f"  Nasal:      {metrics['sectors']['nasal']:6.2f}")
    print(f"  Inferior:   {metrics['sectors']['inferior']:6.2f}")
    print(f"  Temporal:   {metrics['sectors']['temporal']:6.2f}")
    print(f"  σ(pred):    {metrics['pred_std']:6.2f}")
    
    print("="*70)


def print_detailed_results(metrics, fairness=None, dataset_name="Unknown"):
    """Print comprehensive results"""
    print("\n" + "="*70)
    print(f"DETAILED EVALUATION RESULTS - {dataset_name}")
    print("="*70)
    
    print("\n📊 MAIN METRICS:")
    print(f"  MAE:               {metrics['mae']:.2f} μm")
    print(f"  Prediction σ:      {metrics['pred_std']:.2f} μm")
    print(f"  Ground Truth σ:    {metrics['gt_std']:.2f} μm")
    print(f"  Variance Ratio:    {metrics['variance_ratio']:.1%}")
    print(f"  Pearson R:         {metrics['pearson_r']:.3f} (p={metrics['pearson_p']:.2e})")
    
    print("\n📍 SECTOR-WISE MAE:")
    print(f"  Superior:          {metrics['sectors']['superior']:.2f} μm")
    print(f"  Nasal:             {metrics['sectors']['nasal']:.2f} μm")
    print(f"  Inferior:          {metrics['sectors']['inferior']:.2f} μm")
    print(f"  Temporal:          {metrics['sectors']['temporal']:.2f} μm")
    
    if fairness:
        print("\n⚖️  FAIRNESS METRICS:")
        for demo_group, results in fairness.items():
            print(f"\n  {demo_group.capitalize()}:")
            print(f"    Fairness Gap: {results['fairness_gap']:.2f} μm")
            for group, mae in results['group_maes'].items():
                print(f"      {group}: {mae:.2f} μm")
    
    print("="*70)


def compare_to_paper(metrics, dataset_name):
    """Compare computed metrics to paper values"""
    print("\n📋 COMPARISON TO PAPER:")
    
    # Expected values from paper
    expected = {}
    
    if 'fairfedmed' in dataset_name.lower():
        expected = {
            'mae': 19.04,
            'pred_std': 11.8,
            'pearson_r': 0.676
        }
    elif 'grape' in dataset_name.lower():
        expected = {
            'mae': 19.88,
            'pred_std': 22.60
        }
    
    if expected:
        print(f"\nDataset: {dataset_name}")
        for metric_name, expected_value in expected.items():
            actual_value = metrics[metric_name]
            diff = abs(actual_value - expected_value)
            
            # Tolerance
            tolerance = 0.5 if metric_name == 'mae' else (1.0 if metric_name == 'pred_std' else 0.01)
            status = '✓' if diff <= tolerance else '✗'
            
            print(f"  {metric_name}:")
            print(f"    Expected: {expected_value:.2f}")
            print(f"    Actual:   {actual_value:.2f}")
            print(f"    Diff:     {diff:.2f} {status}")


def verify_all_results():
    """Verify all pre-computed results"""
    results_dir = Path('results')
    
    # Main results
    print("\n" + "="*70)
    print("VERIFYING ALL PAPER RESULTS")
    print("="*70)
    
    # Table 1: Main results
    print("\n📊 TABLE 1: Main Results (FairFedMed Test Set)")
    if (results_dir / 'fairfedmed_test_predictions.csv').exists():
        pred, gt = load_predictions(results_dir / 'fairfedmed_test_predictions.csv')
        demo = load_demographics(results_dir / 'fairfedmed_test_demographics.csv')
        
        metrics = compute_metrics(pred, gt)
        fairness = compute_fairness_metrics(pred, gt, demo)
        
        print_table_1_format(metrics, fairness)
        compare_to_paper(metrics, 'fairfedmed')
    else:
        print("  ⚠️  fairfedmed_test_predictions.csv not found")
    
    # Table 2: Cross-modality validation
    print("\n📊 TABLE 2: Cross-Modality Validation (GRAPE)")
    if (results_dir / 'grape_predictions.csv').exists():
        pred, gt = load_predictions(results_dir / 'grape_predictions.csv')
        metrics = compute_metrics(pred, gt)
        
        print_table_2_format(metrics)
        compare_to_paper(metrics, 'grape')
    else:
        print("  ⚠️  grape_predictions.csv not found")
    
    # Baselines comparison
    print("\n📊 BASELINE COMPARISONS:")
    baseline_files = [
        ('mae_baseline_predictions.csv', 'MAE Baseline'),
        ('structured_protocol_predictions.csv', 'Structured Protocol'),
        ('aggressive_baseline_predictions.csv', 'Aggressive Baseline')
    ]
    
    for filename, name in baseline_files:
        filepath = results_dir / 'baselines' / filename
        if filepath.exists():
            pred, gt = load_predictions(filepath)
            metrics = compute_metrics(pred, gt)
            print(f"\n  {name}:")
            print(f"    MAE: {metrics['mae']:.2f} μm")
            print(f"    σ(pred): {metrics['pred_std']:.2f} μm")
        else:
            print(f"\n  {name}: File not found")
    
    print("\n" + "="*70)
    print("✅ Verification complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Verify paper results from pre-computed predictions'
    )
    parser.add_argument('--predictions', type=str,
                        help='Path to predictions CSV file')
    parser.add_argument('--demographics', type=str, default=None,
                        help='Path to demographics CSV file (optional)')
    parser.add_argument('--verify_all', action='store_true',
                        help='Verify all results in results/ directory')
    parser.add_argument('--format', type=str, default='detailed',
                        choices=['detailed', 'table1', 'table2'],
                        help='Output format')
    
    args = parser.parse_args()
    
    if args.verify_all:
        verify_all_results()
        return
    
    if not args.predictions:
        print("Error: Must specify --predictions or use --verify_all")
        parser.print_help()
        return
    
    # Load predictions
    predictions, ground_truth = load_predictions(args.predictions)
    demographics = load_demographics(args.demographics)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, ground_truth)
    fairness = compute_fairness_metrics(predictions, ground_truth, demographics)
    
    # Print results in requested format
    dataset_name = Path(args.predictions).stem
    
    if args.format == 'table1':
        print_table_1_format(metrics, fairness)
    elif args.format == 'table2':
        print_table_2_format(metrics)
    else:
        print_detailed_results(metrics, fairness, dataset_name)
    
    # Compare to paper values
    compare_to_paper(metrics, dataset_name)
    
    print("\n✅ Verification complete!")


if __name__ == '__main__':
    main()
