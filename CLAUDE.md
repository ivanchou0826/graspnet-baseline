# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraspNet-Baseline is a deep learning framework for 6-DoF grasp detection from RGB-D point clouds, the baseline model for the GraspNet-1Billion benchmark (CVPR 2020). This repo also contains a ROS2 node (`graspnet_ros2/`) that wraps the model for real-time robotic manipulation.

## Setup & Installation

```bash
pip install -r requirements.txt

# Compile PointNet2 CUDA ops (required)
cd pointnet2 && python setup.py install && cd ..

# Compile KNN CUDA ops (required)
cd knn && python setup.py install && cd ..

# Install graspnetAPI for evaluation
pip install graspnetAPI
```

Pretrained weights: `checkpoint-rs.tar` (RealSense) and `checkpoint-kn.tar` (Kinect). The realsense model transfers better to new scenes.

## Common Commands

**Train:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense \
  --log_dir logs/log_rs --batch_size 2 \
  --dataset_root /data/Benchmark/graspnet
```

**Evaluate** (set `--collision_thresh -1` for fast inference):
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --checkpoint_path logs/log_rs/checkpoint.tar \
  --dump_dir logs/dump_rs --camera realsense \
  --dataset_root /data/Benchmark/graspnet
```

**Demo on sample data** (`doc/example_data/`):
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py \
  --checkpoint_path checkpoint-rs.tar
```

**Generate tolerance labels** (not included in dataset download):
```bash
cd dataset && bash command_generate_tolerance_label.sh
```

## ROS2 Node

The `graspnet_ros2/` package is a colcon Python package for ROS2 Humble.

**Build and run:**
```bash
# From the humble_ws root
colcon build --packages-select graspnet_ros2
source install/setup.bash

export GRASPNET_ROOT=/path/to/graspnet-baseline
ros2 launch graspnet_ros2 graspnet.launch.py
```

The launch file auto-sets `GRASPNET_ROOT` to the repo root if not already set. To override the checkpoint or any parameter, pass arguments on the command line or edit `graspnet_ros2/launch/graspnet.launch.py`.

**Trigger inference on demand:**
```bash
ros2 service call /graspnet/trigger std_srvs/srv/Trigger {}
```

Inference only runs when the service is called — the node continuously caches the latest synchronized RGB+depth+CameraInfo frame but does not run the GPU model continuously.

**Topics:**
| Topic | Direction | Type | Description |
|---|---|---|---|
| `/camera_2/image` | sub | `sensor_msgs/Image` | RGB input |
| `/camera_2/depth` | sub | `sensor_msgs/Image` | Depth input (16UC1 or 32FC1) |
| `/camera_2/info` | sub | `sensor_msgs/CameraInfo` | Camera intrinsics |
| (configurable) | sub (optional) | `vision_msgs/Detection2DArray` | One or more detection topics from YOLOv8/RT-DETR; set via `det_topics` parameter |
| `/graspnet/visualization` | pub | `sensor_msgs/Image` | BGR image with detection boxes (yellow) and grasp overlays |
| `/graspnet/best_grasp` | pub | `geometry_msgs/PoseStamped` | Highest-score grasp pose (camera frame) |
| `/graspnet/markers` | pub | `visualization_msgs/MarkerArray` | 3D gripper LINE_LIST markers for rviz2 |
| `/graspnet/pointcloud` | pub | `sensor_msgs/PointCloud2` | XYZRGB cloud after plane removal |

**Node parameters** (all settable from launch file):
- `checkpoint_path` — path to `.tar` checkpoint
- `num_point` (20000) — points sampled for inference
- `collision_thresh` (0.01) — set to -1 to skip collision filtering
- `top_k` (50) — maximum grasps before DBSCAN clustering
- `max_depth` (2.0 m) — depth cutoff for point cloud masking
- `remove_plane` (true) — RANSAC plane removal to strip the table
- `plane_dist_thresh` (0.01 m) — RANSAC inlier threshold
- `det_topics` (`['/detections_output']`) — string array of detection topic names; add/remove at runtime via `ros2 param set`
- `det_input_width` / `det_input_height` (0) — detector model input resolution; 0 = same as depth image (no scaling)
- `det_score_thresh` (0.5) — ignore detections below this confidence
- `det_class_filter` ("") — comma-separated class_id whitelist; empty = accept all classes
- `det_timeout_sec` (3.0) — ignore cached detections older than this (seconds)

**Detection ROI flow:**
Each topic in `det_topics` gets its own subscription and cache. At trigger time all non-expired caches are merged: each `Detection2D.bbox` (in detector pixel space) is scaled to depth image resolution (`scale = depth_size / det_input_size`) and unioned into a single binary ROI mask, which is ANDed with the depth validity mask before point cloud extraction. Multiple instances of the same object (multiple bboxes in one topic) and multiple object classes (multiple topics) are both handled this way. Downstream DBSCAN clustering then groups spatially close grasps into one best grasp per object. Each topic's bboxes are drawn in a distinct color on `/graspnet/visualization`.

**Changing detection topics at runtime (no restart needed):**
```bash
# Subscribe to three object-specific topics
ros2 param set /graspnet_node det_topics \
  "['/detections/blue_cube', '/detections/green_cube', '/detections/red_cube']"

# Revert to default single topic
ros2 param set /graspnet_node det_topics "['/detections_output']"
```

## Architecture

### Two-Stage Network (`models/graspnet.py`)

**Stage 1 — `GraspNetStage1`:** PointNet2 backbone (`models/backbone.py`) processes 20,000 input points through 4 Set Abstraction layers, downsampling to 1,024 seed points with 256-dim features. 2 Feature Propagation layers upsample back. `ApproachNet` then predicts objectness and scores 300 candidate approach viewpoints per seed point.

**Stage 2 — `GraspNetStage2`:** `CloudCrop` cylinders (radius=0.05 m) are extracted around the top-scoring seed points. `OperationNet` regresses grasp parameters over a discrete grid: 12 in-plane rotation angles × 4 gripper depths × continuous width. `ToleranceNet` predicts grasp robustness scores on the same grid.

**`pred_decode()`** converts the raw tensor outputs into a flat array of grasp tuples: `(score, width, height, depth, rotation_matrix, center, object_id)` consumed by `GraspGroup` from graspnetAPI.

### Data Flow
```
RGB-D image
  → create_point_cloud_from_depth_image()   [utils/data_utils.py]
  → sample 20k points
  → GraspNetStage1: PointNet2 → viewpoint scores
  → GraspNetStage2: CloudCrop → OperationNet/ToleranceNet → grasp grid
  → pred_decode() → GraspGroup
  → ModelFreeCollisionDetector (optional)
  → nms() + sort_by_score() → top-K grasps
```

### ROS2 Node Post-Processing

After model inference, the node applies:
1. Collision filtering (`ModelFreeCollisionDetector` against the masked point cloud)
2. NMS + score sort → top-K
3. DBSCAN clustering (`eps=0.08 m`) to reduce to one best grasp per object

### Key Files
- `models/graspnet.py` — full network, `pred_decode()`
- `models/backbone.py` — PointNet2 (SA + FP layers)
- `models/modules.py` — ApproachNet, CloudCrop, OperationNet, ToleranceNet
- `models/loss.py` — Objectness (CE) + View (MSE) + Grasp (Huber); grasp loss weighted 0.2×
- `dataset/graspnet_dataset.py` — GraspNetDataset, train/test splits, collision label loading
- `utils/collision_detector.py` — ModelFreeCollisionDetector
- `utils/data_utils.py` — point cloud projection, CameraInfo class
- `utils/loss_utils.py` — grasp parameter constants (`num_view=300`, `num_angle=12`, `num_depth=4`)
- `graspnet_ros2/graspnet_ros2/graspnet_node.py` — ROS2 node (GraspNetNode)
- `graspnet_ros2/launch/graspnet.launch.py` — launch file with parameter defaults

### Dataset Splits
- Train: scenes 0–99
- Test Seen: scenes 100–129 | Test Similar: 130–159 | Test Novel: 160–189

### Demo Data Format (`doc/example_data/`)
- `color.png`, `depth.png`, `workspace_mask.png`
- `meta.mat` with `intrinsic_matrix` and `factor_depth` (scale to convert raw depth to meters)
