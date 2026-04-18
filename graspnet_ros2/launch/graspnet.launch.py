import os
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

# Path to graspnet-baseline repo — override via GRASPNET_ROOT env var before launching
GRASPNET_ROOT = os.environ.get(
    'GRASPNET_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('GRASPNET_ROOT', GRASPNET_ROOT),
        Node(
            package='graspnet_ros2',
            executable='graspnet_node',
            name='graspnet_node',
            output='screen',
            parameters=[{
                'checkpoint_path': os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar'),
                'num_point':        20000,
                'num_view':         300,
                'collision_thresh': 0.01,
                'voxel_size':       0.01,
                'top_k':            50,
            }],
        ),
    ])
