""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--debug_dir', default='', help='If set, save intermediate PLY files (s1/s2/s5) to this directory for inspection in Foxglove/CloudCompare/Open3D')
# ROS2 single-frame mode
parser.add_argument('--from_ros', action='store_true', help='Capture one synchronized frame from ROS2 topics instead of reading files')
parser.add_argument('--rgb_topic',   default='/rgb/camera_3',              help='ROS2 RGB image topic')
parser.add_argument('--depth_topic', default='/camera_3/depth/image_raw',  help='ROS2 depth image topic')
parser.add_argument('--info_topic',  default='/camera_3/depth/camera_info', help='ROS2 CameraInfo topic')
parser.add_argument('--max_depth',   type=float, default=3.0, help='Depth cutoff in metres for ROS2 mode [default: 3.0]')
cfgs = parser.parse_args()


def _save_ply(path, xyz, rgb_float):
    """Save Nx3 xyz + Nx3 rgb [0,1] as a colored PLY via Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb_float, 0, 1).astype(np.float32))
    o3d.io.write_point_cloud(path, pcd)
    print(f'  [debug] {os.path.basename(path)}  {len(xyz)} pts')


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir, debug_dir=''):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # s1: depth-valid points only (before workspace mask)
    depth_mask = depth > 0
    if debug_dir:
        _save_ply(os.path.join(debug_dir, 's1_depth_valid.ply'), cloud[depth_mask], color[depth_mask])

    # s2: after workspace mask
    mask = (workspace_mask.astype(bool) & depth_mask)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    if debug_dir:
        _save_ply(os.path.join(debug_dir, 's2_workspace.ply'), cloud_masked, color_masked)

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # s5: sampled input to model (s3/s4 = plane removal, not used in demo)
    if debug_dir:
        _save_ply(os.path.join(debug_dir, 's5_sampled.ply'), cloud_sampled, color_sampled)

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_and_process_data_from_ros(rgb_topic, depth_topic, info_topic):
    """Capture one synchronized frame from ROS2 and return same format as get_and_process_data()."""
    import rclpy
    import rclpy.node
    import message_filters
    from rclpy.node import Node as RclpyNode
    from sensor_msgs.msg import Image as RosImage, CameraInfo as RosCameraInfo

    captured = {}

    class _OneShot(RclpyNode):
        def __init__(self):
            super().__init__('_graspnet_snap')
            rgb_sub = message_filters.Subscriber(self, RosImage, rgb_topic)
            dep_sub = message_filters.Subscriber(self, RosImage, depth_topic)
            inf_sub = message_filters.Subscriber(self, RosCameraInfo, info_topic)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [rgb_sub, dep_sub, inf_sub], queue_size=5, slop=0.1)
            self._sync.registerCallback(self._cb)
            self._done = False

        def _cb(self, rgb_msg, depth_msg, info_msg):
            captured['rgb']   = rgb_msg
            captured['depth'] = depth_msg
            captured['info']  = info_msg
            self._done = True

    rclpy.init()
    node = _OneShot()
    print(f'Waiting for frame on topics:\n  rgb:   {rgb_topic}\n  depth: {depth_topic}\n  info:  {info_topic}')
    while not node._done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    rgb_msg   = captured['rgb']
    depth_msg = captured['depth']
    info_msg  = captured['info']

    # decode RGB → float32 H×W×3 [0,1]
    enc = rgb_msg.encoding.lower()
    color_u8 = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)[:, :, :3]
    if 'bgr' in enc:
        color_u8 = color_u8[:, :, ::-1].copy()
    color = color_u8.astype(np.float32) / 255.0

    # decode depth
    enc_d = depth_msg.encoding.lower()
    if '16' in enc_d:
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
        factor_depth = 1000.0
    else:
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
        factor_depth = 1.0

    H, W = depth.shape
    print(f'Frame captured: {W}x{H}  depth_enc={depth_msg.encoding}  factor={factor_depth}')

    # resize color to match depth resolution if they differ
    if color.shape[:2] != (H, W):
        from PIL import Image as PilImage
        color_pil = PilImage.fromarray((color * 255).astype(np.uint8))
        color = np.array(color_pil.resize((W, H), PilImage.BILINEAR), dtype=np.float32) / 255.0

    K = info_msg.k
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    camera = CameraInfo(float(W), float(H), fx, fy, cx, cy, factor_depth)
    cloud_org = create_point_cloud_from_depth_image(depth, camera, organized=True)

    depth_mask = (depth > 0) & (depth < cfgs.max_depth * factor_depth)
    cloud_masked = cloud_org[depth_mask]
    color_masked = color[depth_mask]
    print(f'Valid points: {len(cloud_masked)} / {H * W}')

    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    end_points = {
        'point_clouds': torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device),
        'cloud_colors': color_sampled,
    }
    return end_points, cloud_o3d


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    if cfgs.from_ros:
        end_points, cloud = get_and_process_data_from_ros(
            cfgs.rgb_topic, cfgs.depth_topic, cfgs.info_topic)
    else:
        end_points, cloud = get_and_process_data(data_dir, debug_dir=cfgs.debug_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
