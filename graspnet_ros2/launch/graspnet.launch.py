"""
GraspNet ROS2 launch file.

Show all available arguments:
  ros2 launch graspnet_ros2 graspnet.launch.py --show-args

Example:
  ros2 launch graspnet_ros2 graspnet.launch.py \\
    rgb_topic:=/camera_1/image \\
    remove_plane:=true \\
    top_k:=100 \\
    top_k_per_cluster:=5 \\
    nms_rot_thresh_deg:=15.0 \\
    collision_thresh:=-1.0

Trigger inference:
  ros2 service call /graspnet/trigger std_srvs/srv/Trigger {}
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Path to graspnet-baseline repo — override via GRASPNET_ROOT env var before launching
GRASPNET_ROOT = os.environ.get(
    'GRASPNET_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

_ARGS = [
    # ── Camera topics ──────────────────────────────────────────────────────
    ('rgb_topic',           '/camera_1/image',
     '[Camera] RGB image topic (sensor_msgs/Image)'),
    ('depth_topic',         '/camera_1/depth',
     '[Camera] Depth image topic (sensor_msgs/Image, 16UC1 or 32FC1)'),
    ('info_topic',          '/camera_1/info',
     '[Camera] CameraInfo topic (sensor_msgs/CameraInfo)'),

    # ── Model / inference ──────────────────────────────────────────────────
    ('checkpoint_path',     os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar'),
     '[Model] Path to .tar checkpoint file (checkpoint-rs.tar=RealSense, checkpoint-kn.tar=Kinect)'),
    ('num_point',           '20000',
     '[Model] Number of points sampled from depth cloud for inference'),
    ('collision_thresh',    '0.01',
     '[Model] Collision detection threshold in metres; -1.0 to disable (faster inference)'),

    # ── Point cloud pre-processing ─────────────────────────────────────────
    ('max_depth',           '3.0',
     '[PointCloud] Depth cutoff in metres — points farther than this are discarded'),
    ('remove_plane',        'false',
     '[PointCloud] RANSAC plane removal to strip the table surface before inference'),
    ('plane_dist_thresh',   '0.01',
     '[PointCloud] RANSAC inlier threshold in metres for plane removal'),

    # ── Grasp post-processing ──────────────────────────────────────────────
    ('top_k',               '100',
     '[Grasp] Max total grasps kept after DBSCAN+per-cluster selection (final output cap)'),
    ('top_k_per_cluster',   '3',
     '[Grasp] Max grasps kept per DBSCAN cluster (= per object); affects markers and visualization'),
    ('nms_trans_thresh',    '0.03',
     '[Grasp/NMS] Translation threshold in metres — grasps within this distance are deduplicated'),
    ('nms_rot_thresh_deg',  '45.0',
     '[Grasp/NMS] Rotation threshold in degrees — grasps within this angle are deduplicated'),
    ('dbscan_eps',          '0.05',
     '[Grasp/DBSCAN] Cluster radius in metres; grasps within this distance are grouped as one object; -1.0 to disable'),

    # ── Workspace mask ────────────────────────────────────────────────────
    ('workspace_mask_path', '',
     '[Mask] Path to a binary PNG workspace mask; white pixels = keep, black = discard; empty = disabled'),

    # ── Debug ─────────────────────────────────────────────────────────────
    ('debug_pointcloud', 'false',
     '[Debug] Publish intermediate point clouds on /graspnet/debug/* for Foxglove/RViz2 inspection'),

    # ── Detection ROI ──────────────────────────────────────────────────────
    ('det_topics',          "['/detections_output']",
     '[Detection] String array of vision_msgs/Detection2DArray topics; grasps not projecting into any bbox are discarded after inference'),
    ('det_score_thresh',    '0.5',
     '[Detection] Ignore detections below this confidence score'),
    ('det_class_filter',    '',
     '[Detection] Comma-separated class_id whitelist; empty string = accept all classes'),
    ('det_input_width',     '0',
     '[Detection] Detector model input width in pixels; 0 = same as depth image (no scaling)'),
    ('det_input_height',    '0',
     '[Detection] Detector model input height in pixels; 0 = same as depth image (no scaling)'),
    ('det_timeout_sec',     '3.0',
     '[Detection] Ignore cached detections older than this many seconds'),
]


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=desc)
        for name, default, desc in _ARGS
    ]

    node = Node(
        package='graspnet_ros2',
        executable='graspnet_node',
        name='graspnet_node',
        output='screen',
        parameters=[{
            'rgb_topic':          LaunchConfiguration('rgb_topic'),
            'depth_topic':        LaunchConfiguration('depth_topic'),
            'info_topic':         LaunchConfiguration('info_topic'),
            'checkpoint_path':    LaunchConfiguration('checkpoint_path'),
            'num_point':          LaunchConfiguration('num_point'),
            'num_view':           300,
            'collision_thresh':   LaunchConfiguration('collision_thresh'),
            'voxel_size':         0.01,
            'top_k':              LaunchConfiguration('top_k'),
            'top_k_per_cluster':  LaunchConfiguration('top_k_per_cluster'),
            'nms_trans_thresh':   LaunchConfiguration('nms_trans_thresh'),
            'nms_rot_thresh_deg': LaunchConfiguration('nms_rot_thresh_deg'),
            'dbscan_eps':         LaunchConfiguration('dbscan_eps'),
            'max_depth':          LaunchConfiguration('max_depth'),
            'remove_plane':       LaunchConfiguration('remove_plane'),
            'plane_dist_thresh':  LaunchConfiguration('plane_dist_thresh'),
            'workspace_mask_path': LaunchConfiguration('workspace_mask_path'),
            'debug_pointcloud':   LaunchConfiguration('debug_pointcloud'),
            'det_topics':         LaunchConfiguration('det_topics'),
            'det_score_thresh':   LaunchConfiguration('det_score_thresh'),
            'det_class_filter':   LaunchConfiguration('det_class_filter'),
            'det_input_width':    LaunchConfiguration('det_input_width'),
            'det_input_height':   LaunchConfiguration('det_input_height'),
            'det_timeout_sec':    LaunchConfiguration('det_timeout_sec'),
        }],
    )

    return LaunchDescription([
        SetEnvironmentVariable('GRASPNET_ROOT', GRASPNET_ROOT),
        *declared,
        node,
    ])
