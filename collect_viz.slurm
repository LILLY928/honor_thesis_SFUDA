#!/bin/bash
#SBATCH --job-name=collect_viz
#SBATCH --partition=hima
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=log_collect_viz.log
#SBATCH --error=log_collect_viz.err

# Set REPO_DIR to the absolute path of this repo on your cluster.
# Either export it before sbatch, or edit the default below.
REPO_DIR="${REPO_DIR:-$PWD}"

source "$REPO_DIR/venv/bin/activate"
cd "$REPO_DIR"
python collect_viz.py
