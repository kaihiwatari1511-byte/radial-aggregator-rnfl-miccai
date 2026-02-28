"""
scripts/train.py
Main training script for RNFL thickness prediction
"""

import os
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import modules
from models.retfound_rnfl import build_model
from losses.rnfl_losses import get_loss_function
from evaluation.metrics import compute_rnfl_metrics

# Note: Adjust dataloader import based on your specific implementation
try:
    from data.dataloader import create_dataloaders
except ImportError:
    print("Warning: data.dataloader not found. Please implement dataloading logic.")


def parse_args():
    parser = argparse.ArgumentParser(description='Train RNFL prediction model')
    
    # Paths
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (YAML)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    
    # Training
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--accumulation_steps', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    
    # Optional overrides
    parser.add_argument('--lr_encoder', type=float, default=None)
    parser.add_argument('--lr_head', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, 
                accumulation_steps=10):
    """
    Train for one epoch.
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        images = batch['image'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=(scaler is not None)):
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        n_batches += 1
    
    return total_loss / max(1, n_batches)


def validate(model, dataloader, device):
    """
    Validate model and compute metrics.
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            predictions = model(images)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch['target'])
            
            # Collect metadata if available
            for i in range(len(batch['target'])):
                metadata = {
                    'age': batch.get('age', [50] * len(batch['target']))[i],
                    'gender': batch.get('gender_str', ['U'] * len(batch['target']))[i],
                    'race': batch.get('race', [0] * len(batch['target']))[i]
                }
                all_metadata.append(metadata)
    
    if not all_predictions:
        return {'MAE': 999.0, 'Pearson_R': 0.0, 'sigma_pred': 0.0}
        
    # Concatenate all batches
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    metrics = compute_rnfl_metrics(predictions, targets, all_metadata)
    
    return metrics


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config (Simplified - ideally loaded from YAML based on args.config)
    config = {
        'retfound_path': './weights/retfound_weights.pth',  # Anonymized generic path
        'freeze_layers': 0,  # No freezing (aggressive adaptation)
        'dropout': 0.2,
        'loss_type': 'gradient',  # or 'mae', 'pearson', 'structured'
        'lambda_gradient': 0.2,
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config).to(device)
    
    # Get parameter groups for differential learning rates
    encoder_params, head_params = model.get_trainable_parameters()
    
    # Optimizer
    lr_encoder = args.lr_encoder if args.lr_encoder else 5e-5
    lr_head = args.lr_head if args.lr_head else 1e-4
    weight_decay = args.weight_decay if args.weight_decay else 0.10
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': lr_encoder},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=weight_decay)
    
    print(f"  Encoder LR: {lr_encoder}")
    print(f"  Head LR: {lr_head}")
    print(f"  Weight Decay: {weight_decay}")
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    # Loss function
    criterion = get_loss_function(
        loss_type=config['loss_type'],
        lambda_gradient=config.get('lambda_gradient', 0.2)
    )
    
    print(f"  Loss: {config['loss_type']}")
    
    # Mixed precision scaler
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Data loaders
    print("\nLoading data...")
    try:
        train_loader, val_loader, _ = create_dataloaders(
            csv_path=args.data_path,
            tar_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=8,
            img_size=224
        )
    except NameError:
        print("Error: create_dataloaders not imported. Please ensure data.dataloader exists.")
        return
        
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_mae = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=args.accumulation_steps
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MAE: {val_metrics.get('MAE', 999.0):.2f} μm")
        print(f"  Val R: {val_metrics.get('Pearson_R', 0.0):.4f}")
        print(f"  Val σ(pred): {val_metrics.get('sigma_pred', 0.0):.2f} μm")
        print(f"  Fairness Gap: {val_metrics.get('avg_fairness_gap', 0.0):.2f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
        
        # Save best model
        current_mae = val_metrics.get('MAE', float('inf'))
        if current_mae < best_mae:
            best_mae = current_mae
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae,
                'metrics': val_metrics
            }
            
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (MAE: {best_mae:.2f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping triggered (patience: {args.patience})")
            break
    
    # Save final results
    results = {
        'best_mae': best_mae,
        'history': history,
        'config': config
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Best MAE: {best_mae:.2f} μm")
    print(f"  Checkpoints saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
