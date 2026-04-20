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


def generate_launch_description():
    return LaunchDescription([
        # ---- optional launch arguments ----
        # Override detection topics at launch time, e.g.:
        #   ros2 launch graspnet_ros2 graspnet.launch.py \
        #     det_topics:="['/detections/blue_cube', '/detections/green_cube']"
        DeclareLaunchArgument(
            'det_topics',
            default_value="['/detections_output']",
            description='String array of vision_msgs/Detection2DArray topics to subscribe to'),
        DeclareLaunchArgument(
            'det_score_thresh',
            default_value='0.5',
            description='Minimum detection confidence score'),
        DeclareLaunchArgument(
            'det_class_filter',
            default_value='',
            description='Comma-separated class_id whitelist (empty = all classes)'),
        DeclareLaunchArgument(
            'det_input_width',
            default_value='0',
            description='Detector model input width in pixels (0 = same as depth image)'),
        DeclareLaunchArgument(
            'det_input_height',
            default_value='0',
            description='Detector model input height in pixels (0 = same as depth image)'),

        SetEnvironmentVariable('GRASPNET_ROOT', GRASPNET_ROOT),
        Node(
            package='graspnet_ros2',
            executable='graspnet_node',
            name='graspnet_node',
            output='screen',
            parameters=[{
                'checkpoint_path':  os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar'),
                'num_point':        20000,
                'num_view':         300,
                'collision_thresh': 0.01,
                'voxel_size':       0.01,
                'top_k':            50,
                'max_depth':        2.0,
                'remove_plane':     True,
                'plane_dist_thresh': 0.01,
                'det_topics':        LaunchConfiguration('det_topics'),
                'det_score_thresh':  LaunchConfiguration('det_score_thresh'),
                'det_class_filter':  LaunchConfiguration('det_class_filter'),
                'det_input_width':   LaunchConfiguration('det_input_width'),
                'det_input_height':  LaunchConfiguration('det_input_height'),
            }],
        ),
    ])
