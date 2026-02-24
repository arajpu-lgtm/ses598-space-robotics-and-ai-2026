from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    pkg_share = FindPackageShare('cart_pole_optimal_control').find('cart_pole_optimal_control')
    urdf_model_path = os.path.join(pkg_share, 'models', 'cart_pole', 'model.urdf')
    rviz_config_path = os.path.join(pkg_share, 'config', 'cart_pole.rviz')

    # Tunable LQR params
    controller_q_diag = LaunchConfiguration('controller_q_diag')
    controller_r = LaunchConfiguration('controller_r')
    use_rviz = LaunchConfiguration('use_rviz')

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        'controller_q_diag',
        default_value='[1.0, 1.0, 1.0, 1.0]',
        description='Diagonal of Q for LQR: [x, x_dot, theta, theta_dot]'
    ))
    ld.add_action(DeclareLaunchArgument(
        'controller_r',
        default_value='0.1',
        description='Scalar R for LQR'
    ))
    ld.add_action(DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz (false for faster optimizer runs)'
    ))

    # Gazebo (headless sim server)
    ld.add_action(ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', 'empty.sdf'],
        output='screen'
    ))

    # Robot state publisher provides robot_description topic for spawn
    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['cat ', urdf_model_path]),
            'publish_frequency': 50.0,
            'use_tf_static': True,
            'ignore_timestamp': True
        }]
    ))

    # Spawn robot in Gazebo using robot_description topic
    ld.add_action(Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'cart_pole',
            '-allow_renaming', 'true'
        ],
        output='screen'
    ))

    # Bridge topics between Gazebo and ROS2
    ld.add_action(Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge',
        output='screen',
        arguments=[
            # ROS -> Gazebo force command
            '/model/cart_pole/joint/cart_to_base/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
            # Gazebo -> ROS joint state
            '/world/empty/model/cart_pole/joint_state@sensor_msgs/msg/JointState[ignition.msgs.Model',
            # Gazebo -> ROS clock
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'
        ],
    ))

    # Republish state (your package)
    ld.add_action(Node(
        package='cart_pole_optimal_control',
        executable='state_republisher',
        name='state_republisher',
        output='screen'
    ))

    # Force visualizer (your package)
    ld.add_action(Node(
        package='cart_pole_optimal_control',
        executable='force_visualizer',
        name='force_visualizer',
        output='screen'
    ))

    # LQR Controller (tunable via launch args)
    ld.add_action(Node(
        package='cart_pole_optimal_control',
        executable='lqr_controller',
        name='lqr_controller',
        output='screen',
        parameters=[{
            'q_diag': controller_q_diag,
            'r': controller_r
        }]
    ))

    # Earthquake generator
    ld.add_action(Node(
        package='cart_pole_optimal_control',
        executable='earthquake_force_generator',
        name='earthquake_force_generator',
        output='screen',
        parameters=[{
            'base_amplitude': 15.0,
            'frequency_range': [0.5, 4.0],
            'update_rate': 50.0
        }]
    ))

    # RViz (optional)
    ld.add_action(Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[{'update_rate': 50.0}]
    ))

    return ld