#!/bin/bash
# =============================================================
# AFM 3D Molecular Reconstruction - Training Scripts
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)/micro"

CONFIG="$SCRIPT_DIR/configs/config.json"
SAVE_DIR="$SCRIPT_DIR/checkpoints"

echo "============================================"
echo "  AFM 3D Molecular Reconstruction Pipeline"
echo "  Config: $CONFIG"
echo "============================================"

# ---- 1. Train Main Model (Video ViT + Diffusion) ----
echo ""
echo "[1/3] Training Video ViT + Conditional Diffusion Model..."
python3 -m src.train --config "$CONFIG"

# ---- 2. Train Baseline (3D-ResNet) ----
# To train resnet3d, change "model_type" in config.json to "resnet3d" and run again,
# or uncomment below with a separate config file:
# echo ""
# echo "[2/3] Training 3D-ResNet Baseline..."
# python3 -m src.train --config "$SCRIPT_DIR/config_resnet3d.json"

# ---- 3. Visualization: Training Curves ----
echo ""
echo "[3/5] Generating training curve visualizations..."
python3 -c "
import sys; sys.path.insert(0, 'micro')
from src.utils.visualize import plot_training_curves
import os

save_dir = '$SAVE_DIR'
vis_dir = 'micro/visualizations'
os.makedirs(vis_dir, exist_ok=True)

for model_type in ['diffusion', 'resnet3d']:
    hist = os.path.join(save_dir, f'history_{model_type}.json')
    if os.path.exists(hist):
        plot_training_curves(hist, os.path.join(vis_dir, f'curves_{model_type}.png'))
        print(f'  ✓ Saved training curves: visualizations/curves_{model_type}.png')
"

# ---- 4. Visualization: Molecule 3D Comparisons ----
echo ""
echo "[4/5] Generating molecule 3D visualizations..."
if [ -f "$SAVE_DIR/best_diffusion.pt" ]; then
    python3 -m src.visualize_val \
        --checkpoint "$SAVE_DIR/best_diffusion.pt" \
        --num_samples 10 \
        --output_dir micro/visualizations/molecules_diffusion
    echo "  ✓ Saved 10 molecule visualizations for diffusion model"
fi

if [ -f "$SAVE_DIR/best_resnet3d.pt" ]; then
    python3 -m src.visualize_val \
        --checkpoint "$SAVE_DIR/best_resnet3d.pt" \
        --num_samples 10 \
        --output_dir micro/visualizations/molecules_resnet3d
    echo "  ✓ Saved 10 molecule visualizations for resnet3d model"
fi

# ---- 5. Generate Summary Report ----
echo ""
echo "[5/5] Generating summary report..."
python3 -c "
import sys; sys.path.insert(0, 'micro')
import os
import json
from datetime import datetime

save_dir = '$SAVE_DIR'
vis_dir = 'micro/visualizations'
report_path = os.path.join(save_dir, 'training_summary.txt')

with open(report_path, 'w') as f:
    f.write('='*70 + '\n')
    f.write('AFM 3D Molecular Reconstruction - Training Summary\n')
    f.write('='*70 + '\n')
    f.write(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n')
    f.write('\n')

    # Check which models were trained
    models_trained = []
    if os.path.exists(os.path.join(save_dir, 'best_diffusion.pt')):
        models_trained.append('diffusion')
    if os.path.exists(os.path.join(save_dir, 'best_resnet3d.pt')):
        models_trained.append('resnet3d')

    f.write(f'Models Trained: {', '.join(models_trained) if models_trained else 'None'}\n')
    f.write('\n')

    # List all output files
    f.write('OUTPUT FILES:\n')
    f.write('-'*70 + '\n')

    for model_type in models_trained:
        f.write(f'\n[{model_type.upper()} Model]\n\n')

        # Model checkpoint
        ckpt = os.path.join(save_dir, f'best_{model_type}.pt')
        if os.path.exists(ckpt):
            size_mb = os.path.getsize(ckpt) / 1024 / 1024
            f.write(f'✓ Model Checkpoint ({size_mb:.1f} MB):\n')
            f.write(f'  {os.path.abspath(ckpt)}\n\n')

        # Training history
        hist = os.path.join(save_dir, f'history_{model_type}.json')
        if os.path.exists(hist):
            f.write(f'✓ Training History:\n')
            f.write(f'  {os.path.abspath(hist)}\n\n')

        # Evaluation metrics
        metrics = os.path.join(save_dir, f'metrics_{model_type}.json')
        if os.path.exists(metrics):
            f.write(f'✓ Evaluation Metrics (6 dimensions):\n')
            f.write(f'  {os.path.abspath(metrics)}\n')
            with open(metrics, 'r') as mf:
                data = json.load(mf)
                if data:
                    last = data[-1]
                    f.write(f'  - RMSD: {last.get(\"rmsd_mean\", 0):.4f}\n')
                    f.write(f'  - Bottom Recall: {last.get(\"bottom_recall_mean\", 0):.4f}\n')
                    f.write(f'  - Bottom RMSD: {last.get(\"bottom_rmsd_mean\", 0):.4f}\n')
                    f.write(f'  - Bond Validity: {last.get(\"bond_validity_mean\", 0):.4f}\n')
                    f.write(f'  - Count Accuracy: {last.get(\"count_exact_match\", 0):.4f}\n')
                    f.write(f'  - Composite Score: {last.get(\"composite_score\", 0):.4f}\n')
            f.write('\n')

        # Model predictions (4 required fields)
        pred = os.path.join(save_dir, f'predictions_{model_type}.json')
        if os.path.exists(pred):
            f.write(f'✓ Model Predictions (4 required fields):\n')
            f.write(f'  {os.path.abspath(pred)}\n')
            f.write(f'  Contains: coords, atom_types, n_atoms_pred, retrieval_cid_indices\n\n')

        # Training curves
        curve = os.path.join(vis_dir, f'curves_{model_type}.png')
        if os.path.exists(curve):
            f.write(f'✓ Training Curves Visualization:\n')
            f.write(f'  {os.path.abspath(curve)}\n\n')

        # Molecule visualizations
        mol_dir = os.path.join(vis_dir, f'molecules_{model_type}')
        if os.path.exists(mol_dir):
            mol_files = [f for f in os.listdir(mol_dir) if f.endswith('.png')]
            f.write(f'✓ Molecule 3D Visualizations ({len(mol_files)} samples):\n')
            f.write(f'  {os.path.abspath(mol_dir)}/\n\n')

    # Training logs
    log_file = os.path.join(save_dir, 'training.log')
    if os.path.exists(log_file):
        f.write(f'✓ Training Log:\n')
        f.write(f'  {os.path.abspath(log_file)}\n\n')

    f.write('='*70 + '\n')

print(f'  ✓ Summary report saved: {report_path}')
"

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo ""

# Print summary
python3 -c "
import sys; sys.path.insert(0, 'micro')
import os

save_dir = '$SAVE_DIR'
report_path = os.path.join(save_dir, 'training_summary.txt')

if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        print(f.read())
else:
    print('Summary report not found.')
"
