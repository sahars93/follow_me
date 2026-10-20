#!/usr/bin/env python3
"""UWB follow-me baseline using proportional-derivative control."""

import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Float32


class PDFollowController(Node):
    """Control following distance and heading from two UWB observations."""

    def __init__(self):
        super().__init__('pd_controller')

        self.declare_parameter('inference_rate', 50.0)
        self.declare_parameter('desired_distance', 1.5)
        self.declare_parameter('kp_linear', 0.8)
        self.declare_parameter('kd_linear', 0.15)
        self.declare_parameter('kp_angular', 1.5)
        self.declare_parameter('kd_angular', 0.2)
        self.declare_parameter('max_linear_speed', 1.2)
        self.declare_parameter('max_angular_speed', 1.5)
        self.declare_parameter('distance_deadband', 0.05)
        self.declare_parameter('heading_deadband', 0.03)
        self.declare_parameter('require_leader_motion', True)
        self.declare_parameter('leader_odom_topic', '/tb3_2/odom')
        self.declare_parameter('leader_linear_threshold', 0.01)
        self.declare_parameter('leader_angular_threshold', 0.01)
        self.declare_parameter('cmd_vel_topic', '/tb3_1/cmd_vel')
        self.declare_parameter('anchor1_range_topic', '/anchor1/tag/range')
        self.declare_parameter('anchor2_range_topic', '/anchor2/tag/range')
        self.declare_parameter('anchor1_direction_topic', '/anchor1/tag/direction')
        self.declare_parameter('anchor2_direction_topic', '/anchor2/tag/direction')

        rate = float(self.get_parameter('inference_rate').value)
        if rate <= 0.0:
            raise ValueError('inference_rate must be greater than zero')

        self.desired_distance = float(
            self.get_parameter('desired_distance').value)
        self.kp_linear = float(self.get_parameter('kp_linear').value)
        self.kd_linear = float(self.get_parameter('kd_linear').value)
        self.kp_angular = float(self.get_parameter('kp_angular').value)
        self.kd_angular = float(self.get_parameter('kd_angular').value)
        self.max_linear_speed = float(
            self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(
            self.get_parameter('max_angular_speed').value)
        self.distance_deadband = float(
            self.get_parameter('distance_deadband').value)
        self.heading_deadband = float(
            self.get_parameter('heading_deadband').value)
        self.leader_linear_threshold = float(
            self.get_parameter('leader_linear_threshold').value)
        self.leader_angular_threshold = float(
            self.get_parameter('leader_angular_threshold').value)
        require_leader_motion = bool(
            self.get_parameter('require_leader_motion').value)
        self.policy_started = not require_leader_motion

        self.latest_data = [0.0, 0.0, 0.0, 0.0]
        self.received = [False] * 4
        self.previous_distance_error = None
        self.previous_heading_error = None
        self.previous_time = None

        topics = [
            ('anchor1_range_topic', Range, 0, 'range'),
            ('anchor2_range_topic', Range, 1, 'range'),
            ('anchor1_direction_topic', Float32, 2, 'data'),
            ('anchor2_direction_topic', Float32, 3, 'data'),
        ]
        self._uwb_subscriptions = [
            self.create_subscription(
                msg_type,
                self._string_param(parameter),
                self._sensor_callback(index, field),
                10,
            )
            for parameter, msg_type, index, field in topics
        ]
        self._leader_subscription = self.create_subscription(
            Odometry,
            self._string_param('leader_odom_topic'),
            self.leader_odom_callback,
            10,
        )
        self.cmd_vel_publisher = self.create_publisher(
            Twist, self._string_param('cmd_vel_topic'), 10)
        self.timer = self.create_timer(1.0 / rate, self.control_step)

        self.get_logger().info(
            f'PD baseline started at {rate:.1f} Hz; desired distance '
            f'{self.desired_distance:.2f} m')

    def _string_param(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def _sensor_callback(self, index, field):
        def callback(msg):
            self.latest_data[index] = float(getattr(msg, field))
            self.received[index] = True
        return callback

    def leader_odom_callback(self, msg):
        """Latch controller execution after the leader first moves."""
        if self.policy_started:
            return
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular
        linear_speed = math.sqrt(
            linear.x ** 2 + linear.y ** 2 + linear.z ** 2)
        angular_speed = math.sqrt(
            angular.x ** 2 + angular.y ** 2 + angular.z ** 2)
        if (
            linear_speed > self.leader_linear_threshold
            or angular_speed > self.leader_angular_threshold
        ):
            self.policy_started = True
            self.get_logger().info(
                'Leader motion detected; PD controller is now active')

    @staticmethod
    def mean_heading(first_degrees, second_degrees):
        """Return the circular mean of the two direction measurements."""
        first = math.radians(first_degrees)
        second = math.radians(second_degrees)
        return math.atan2(
            math.sin(first) + math.sin(second),
            math.cos(first) + math.cos(second),
        )

    def control_step(self):
        if not all(self.received):
            return
        if not self.policy_started:
            self.cmd_vel_publisher.publish(Twist())
            return

        now = self.get_clock().now()
        distance = 0.5 * (self.latest_data[0] + self.latest_data[1])
        heading = self.mean_heading(self.latest_data[2], self.latest_data[3])
        distance_error = distance - self.desired_distance
        heading_error = heading

        distance_derivative = 0.0
        heading_derivative = 0.0
        if self.previous_time is not None:
            dt = (now - self.previous_time).nanoseconds / 1e9
            if dt > 0.0:
                distance_derivative = (
                    distance_error - self.previous_distance_error) / dt
                heading_delta = math.atan2(
                    math.sin(heading_error - self.previous_heading_error),
                    math.cos(heading_error - self.previous_heading_error),
                )
                heading_derivative = heading_delta / dt

        self.previous_time = now
        self.previous_distance_error = distance_error
        self.previous_heading_error = heading_error

        if abs(distance_error) <= self.distance_deadband:
            distance_error = 0.0
            distance_derivative = 0.0
        if abs(heading_error) <= self.heading_deadband:
            heading_error = 0.0
            heading_derivative = 0.0

        linear = (
            self.kp_linear * distance_error
            + self.kd_linear * distance_derivative
        )
        angular = (
            self.kp_angular * heading_error
            + self.kd_angular * heading_derivative
        )

        command = Twist()
        command.linear.x = max(
            0.0, min(self.max_linear_speed, linear))
        command.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular),
        )
        self.cmd_vel_publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = PDFollowController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_vel_publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
