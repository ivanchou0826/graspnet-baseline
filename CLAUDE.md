# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraspNet-Baseline is a deep learning framework for 6-DoF grasp detection from RGB-D point clouds, the baseline model for the GraspNet-1Billion benchmark (CVPR 2020). It predicts grasp poses and tolerance/robustness scores for robotic manipulation.

## Setup & Installation

```bash
pip install -r requirements.txt

# Compile PointNet2 CUDA ops (required)
cd pointnet2
python setup.py install
cd ..

# Compile KNN CUDA ops (required)
cd knn
python setup.py install
cd ..

# Install graspnetAPI for evaluation
pip install graspnetAPI
```

Camera options: `realsense` or `kinect`.

## Common Commands

**Train:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense \
  --log_dir logs/log_rs --batch_size 2 \
  --dataset_root /data/Benchmark/graspnet
```

**Evaluate:**
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --checkpoint_path logs/log_rs/checkpoint.tar \
  --dump_dir logs/dump_rs --camera realsense \
  --dataset_root /data/Benchmark/graspnet
```

**Run demo on sample data:**
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --checkpoint_path logs/log_kn/checkpoint.tar
```

**Generate tolerance labels:**
```bash
cd dataset && bash command_generate_tolerance_label.sh
```

## Architecture

### Two-Stage Network (`models/graspnet.py`)

**Stage 1 — `GraspNetStage1`:** PointNet2 backbone (`models/backbone.py`) extracts hierarchical features from the input point cloud (20,000 points → 1,024 seed points with 256-dim features). `ApproachNet` predicts objectness and scores 300 candidate viewpoints per seed point.

**Stage 2 — `GraspNetStage2`:** `CloudCrop` extracts cylindrical local patches (radius=0.05m) around high-scoring seed points. `OperationNet` regresses grasp parameters: 12 in-plane angles × 4 depths × width. `ToleranceNet` predicts grasp robustness.

**`pred_decode()`** converts raw network outputs into final grasp representations: (score, width, height, depth, rotation_matrix, center, object_id).

### Data Flow
```
RGB-D → Point Cloud → PointNet2 → Stage1: viewpoint scores
                                → Stage2: grasp params (angle/width/depth/tolerance)
                                → Collision filtering → Grasp output
```

### Key Modules
- `models/backbone.py` — PointNet2 with 4 SA layers + 2 FP layers
- `models/modules.py` — ApproachNet, CloudCrop, OperationNet, ToleranceNet
- `models/loss.py` — Objectness (CE) + View (MSE) + Grasp (Huber) losses; grasp loss weighted at 0.2×
- `dataset/graspnet_dataset.py` — GraspNetDataset, train/test splits, collision labels
- `utils/collision_detector.py` — ModelFreeCollisionDetector for post-processing
- `utils/data_utils.py` — Point cloud processing, camera projection utilities
- `utils/loss_utils.py` — Grasp parameter constants (num_view=300, num_angle=12, num_depth=4)

### Dataset Splits
- Train: scenes 0–99
- Test Seen: scenes 100–129
- Test Similar: scenes 130–159
- Test Novel: scenes 160–189

### Demo Data Format (`doc/example_data/`)
- `color.png`, `depth.png`, `workspace_mask.png`
- `meta.mat` with `intrinsic_matrix` and `factor_depth`
