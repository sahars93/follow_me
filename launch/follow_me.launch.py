"""Launch the simulated UWB sensor and follow-me controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    controller = LaunchConfiguration('controller')
    controller_parameters = [{
        'cmd_vel_topic': LaunchConfiguration('cmd_vel'),
        'leader_odom_topic': LaunchConfiguration('leader_odom'),
        'model_path': LaunchConfiguration('model_path'),
        'inference_rate': LaunchConfiguration('inference_rate'),
        'max_linear_speed': LaunchConfiguration('max_linear_speed'),
        'require_leader_motion': LaunchConfiguration('require_leader_motion'),
    }]
    pd_parameters = [{
        name: value
        for name, value in controller_parameters[0].items()
        if name != 'model_path'
    }]

    def controller_node(selection, executable):
        return Node(
            package='follow_me',
            executable=executable,
            name=f'{selection}_controller',
            output='screen',
            parameters=(
                pd_parameters if selection == 'pd' else controller_parameters),
            condition=IfCondition(PythonExpression([
                "'", controller, "' == '", selection, "'",
            ])),
        )

    recorder_node = Node(
        package='follow_me',
        executable='trajectory_recorder.py',
        name='trajectory_recorder',
        output='screen',
        parameters=[{
            'controller': controller,
            'leader_odom_topic': LaunchConfiguration('leader_odom'),
            'follower_odom_topic': LaunchConfiguration('follower_odom'),
            'csv_output_dir': LaunchConfiguration('csv_output_dir'),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'controller',
            default_value='mlp',
            choices=['mlp', 'svm', 'lstm', 'pd'],
            description='Learned policy or PD baseline to run'),
        DeclareLaunchArgument('follower_odom', default_value='/tb3_1/odom'),
        DeclareLaunchArgument('leader_odom', default_value='/tb3_2/odom'),
        DeclareLaunchArgument('cmd_vel', default_value='/tb3_1/cmd_vel'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('inference_rate', default_value='50.0'),
        DeclareLaunchArgument('max_linear_speed', default_value='1.2'),
        DeclareLaunchArgument('require_leader_motion', default_value='true'),
        DeclareLaunchArgument(
            'csv_output_dir',
            default_value='/home/sahar/workspaces/follow_me_ros2/src/follow_me_ros2/imgs'),
        DeclareLaunchArgument('range_noise_stddev', default_value='0.02'),
        DeclareLaunchArgument('direction_noise_stddev', default_value='1.0'),
        DeclareLaunchArgument('noise_seed', default_value='42'),
        Node(
            package='follow_me',
            executable='gazebo_uwb_publisher.py',
            name='gazebo_uwb_publisher',
            output='screen',
            parameters=[{
                'follower_odom_topic': LaunchConfiguration('follower_odom'),
                'leader_odom_topic': LaunchConfiguration('leader_odom'),
                'range_noise_stddev': LaunchConfiguration(
                    'range_noise_stddev'),
                'direction_noise_stddev': LaunchConfiguration(
                    'direction_noise_stddev'),
                'noise_seed': LaunchConfiguration('noise_seed'),
            }],
        ),
        controller_node('mlp', 'mlp_controller.py'),
        controller_node('svm', 'SVM_main_uwb.py'),
        controller_node('lstm', 'LSTM_main_uwb.py'),
        controller_node('pd', 'pd_controller.py'),
        recorder_node,
    ])
