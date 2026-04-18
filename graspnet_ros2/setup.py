from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'graspnet_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ivan.chou',
    maintainer_email='sinacloud@gmail.com',
    description='GraspNet-1Billion ROS2 node for 6-DoF grasp prediction',
    license='MIT',
    entry_points={
        'console_scripts': [
            'graspnet_node = graspnet_ros2.graspnet_node:main',
        ],
    },
)
