#!/usr/bin/env python3
"""Record leader and follower odometry into a controller-specific CSV file."""

from csv import writer
import os

from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class TrajectoryRecorder(Node):
    """Collect leader and follower odometry samples for later plotting."""

    def __init__(self):
        super().__init__('trajectory_recorder')
        self.declare_parameter('controller', 'mlp')
        self.declare_parameter(
            'csv_output_dir',
            '/home/sahar/workspaces/follow_me_ros2/src/follow_me_ros2/imgs')
        self.declare_parameter('leader_odom_topic', '/tb3_2/odom')
        self.declare_parameter('follower_odom_topic', '/tb3_1/odom')

        self.controller = self._string_param('controller')
        self.csv_output_dir = self._string_param('csv_output_dir')
        self.output_path = os.path.join(
            self.csv_output_dir, f'follow_me_{self.controller}.csv')

        self.start_time = self.get_clock().now()
        self.rows = []

        self._leader_subscription = self.create_subscription(
            Odometry,
            self._string_param('leader_odom_topic'),
            self._leader_callback,
            10,
        )
        self._follower_subscription = self.create_subscription(
            Odometry,
            self._string_param('follower_odom_topic'),
            self._follower_callback,
            10,
        )

        self.get_logger().info(
            f'Recording odometry for {self.controller!r} to {self.output_path}')

    def _string_param(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def _record(self, source, msg):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular
        self.rows.append([
            self.controller,
            source,
            f'{elapsed:.6f}',
            f'{msg.pose.pose.position.x:.6f}',
            f'{msg.pose.pose.position.y:.6f}',
            f'{(linear.x ** 2 + linear.y ** 2) ** 0.5:.6f}',
            f'{angular.z:.6f}',
        ])

    def _leader_callback(self, msg):
        self._record('leader', msg)

    def _follower_callback(self, msg):
        self._record('follower', msg)

    def save_csv(self):
        os.makedirs(self.csv_output_dir, exist_ok=True)
        with open(self.output_path, 'w', newline='', encoding='utf-8') as file:
            csv_writer = writer(file)
            csv_writer.writerow([
                'controller',
                'source',
                'time_s',
                'x_m',
                'y_m',
                'linear_mps',
                'angular_rps',
            ])
            csv_writer.writerows(self.rows)
        self.get_logger().info(f'Saved CSV log to {self.output_path}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save_csv()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()