"""Запуск резервной одометрии трамвая.

    ros2 launch tram_backup_odometry tram_backup_odometry.launch.py
    ros2 launch tram_backup_odometry tram_backup_odometry.launch.py params_file:=/path/my.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare('tram_backup_odometry')
    params_file = LaunchConfiguration('params_file')
    log_level = LaunchConfiguration('log_level')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([share, 'config', 'tram_backup_odometry.yaml']),
            description='YAML с параметрами ноды (топики, карта, GNSS-старт, геометрия)'),
        DeclareLaunchArgument('log_level', default_value='info',
                              description='debug | info | warn | error'),
        Node(
            package='tram_backup_odometry',
            executable='node',
            name='tram_backup_odometry',
            output='screen',
            emulate_tty=True,
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
            # страховка: если процесс всё же завершится аварийно — перезапуск
            respawn=True,
            respawn_delay=0.5,
        ),
    ])
