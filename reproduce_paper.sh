#!/bin/bash
# reproduce_paper.sh
# One-click reproduction of all MICCAI paper results
# Estimated time: 10-12 hours on A100 GPU

set -e  # Exit on error

echo "=========================================="
echo "MICCAI 2026 Paper Reproduction"
echo "=========================================="
echo ""

# Configuration
DATA_PATH="${1:-./data/FairFedMed}"
OUTPUT_DIR="${2:-./results}"
GPU_ID="${3:-0}"

# Create output directories
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/tables
mkdir -p $OUTPUT_DIR/figures

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  GPU ID: $GPU_ID"
echo ""

# ============================================================================
# STEP 1: Train Baseline Models
# ============================================================================
echo "[1/5] Training Baseline Models..."
echo "----------------------------------------"

# Naive baseline (LR=1e-4, no freeze)
echo "  [1a] Training Naive Baseline..."
python scripts/train.py \
    --config configs/baseline.yaml \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR/checkpoints/naive \
    --experiment_name "naive" \
    > $OUTPUT_DIR/logs/naive.log 2>&1

# Aggressive baseline (LR=5e-5, no freeze)
echo "  [1b] Training Aggressive Baseline..."
python scripts/train.py \
    --config configs/aggressive.yaml \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR/checkpoints/aggressive \
    --experiment_name "aggressive" \
    > $OUTPUT_DIR/logs/aggressive.log 2>&1

# Structured protocol (freeze_50%, Pearson loss)
echo "  [1c] Training Structured Protocol..."
python scripts/train.py \
    --config configs/structured.yaml \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR/checkpoints/structured \
    --experiment_name "structured" \
    > $OUTPUT_DIR/logs/structured.log 2>&1

echo "  ✓ Baselines complete!"
echo ""

# ============================================================================
# STEP 2: Train Our Gradient-Loss Model
# ============================================================================
echo "[2/5] Training Gradient-Loss Model (Our Method)..."
echo "----------------------------------------"

python scripts/train.py \
    --config configs/gradient_loss.yaml \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR/checkpoints/gradient_loss \
    --experiment_name "gradient_loss" \
    > $OUTPUT_DIR/logs/gradient_loss.log 2>&1

echo "  ✓ Gradient-Loss training complete!"
echo ""

# ============================================================================
# STEP 3: Run Ablation Studies
# ============================================================================
echo "[3/5] Running Ablation Studies..."
echo "----------------------------------------"

# Freezing ablation
echo "  [3a] Freezing Strategy Ablation..."
python scripts/ablation.py \
    --study freezing \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR/ablations/freezing \
    > $OUTPUT_DIR/logs/ablation_freezing.log 2>&1

# Loss complexity ablation
echo "  [3b] Loss Complexity Ablation..."
python scripts/ablation.py \
    --study loss_complexity \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR/ablations/loss_complexity \
    > $OUTPUT_DIR/logs/ablation_loss.log 2>&1

echo "  ✓ Ablations complete!"
echo ""

# ============================================================================
# STEP 4: Evaluate All Models
# ============================================================================
echo "[4/5] Evaluating Models on Test Set..."
echo "----------------------------------------"

for model in naive aggressive structured gradient_loss; do
    echo "  Evaluating $model..."
    python scripts/evaluate.py \
        --checkpoint $OUTPUT_DIR/checkpoints/$model/best_model.pth \
        --data_path $DATA_PATH \
        --split test \
        --output $OUTPUT_DIR/results_${model}.csv \
        > $OUTPUT_DIR/logs/eval_${model}.log 2>&1
done

echo "  ✓ Evaluation complete!"
echo ""

# ============================================================================
# STEP 5: Generate Paper Tables and Figures
# ============================================================================
echo "[5/5] Generating Tables and Figures..."
echo "----------------------------------------"

# Generate Table 1 (Multi-backbone comparison)
echo "  Generating Table 1..."
python scripts/generate_tables.py \
    --table 1 \
    --results_dir $OUTPUT_DIR \
    --output $OUTPUT_DIR/tables/table1.csv

# Generate Table 2 (Ablation study)
echo "  Generating Table 2..."
python scripts/generate_tables.py \
    --table 2 \
    --results_dir $OUTPUT_DIR/ablations \
    --output $OUTPUT_DIR/tables/table2.csv

# Generate Figure 2 (Template overfitting visualization)
echo "  Generating Figure 2..."
python scripts/generate_figure2.py \
    --gradient_checkpoint $OUTPUT_DIR/checkpoints/gradient_loss/best_model.pth \
    --structured_checkpoint $OUTPUT_DIR/checkpoints/structured/best_model.pth \
    --data_path $DATA_PATH \
    --output $OUTPUT_DIR/figures/figure2.png

echo "  ✓ Tables and figures complete!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=========================================="
echo "REPRODUCTION COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key Files:"
echo "  Tables: $OUTPUT_DIR/tables/"
echo "  Figures: $OUTPUT_DIR/figures/"
echo "  Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  Logs: $OUTPUT_DIR/logs/"
echo ""
echo "Main Results:"
echo "  - Table 1: Multi-backbone comparison"
echo "  - Table 2: Ablation studies"
echo "  - Figure 2: Template overfitting visualization"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/tables/table1.csv"
echo "  cat $OUTPUT_DIR/tables/table2.csv"
echo "  cat $OUTPUT_DIR/figures/figure2.png (Requires image viewer)"
echo ""
echo "=========================================="

# Optional: Print summary statistics
echo ""
echo "Quick Summary:"
echo "----------------------------------------"
python scripts/summarize_results.py --results_dir $OUTPUT_DIR
