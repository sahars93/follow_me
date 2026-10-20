#!/usr/bin/env python3
"""Generate two simulated UWB observations from Gazebo odometry."""

import math
import random

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Float32


class SimUWB(Node):

    def __init__(self):
        super().__init__('gazebo_uwb_publisher')

        self.declare_parameter('anchor_offset', 0.2)
        self.declare_parameter('range_noise_stddev', 0.02)
        self.declare_parameter('direction_noise_stddev', 1.0)
        self.declare_parameter('noise_seed', 42)
        self.declare_parameter('follower_odom_topic', '/tb3_1/odom')
        self.declare_parameter('leader_odom_topic', '/tb3_2/odom')
        self.declare_parameter('anchor1_range_topic', '/anchor1/tag/range')
        self.declare_parameter('anchor2_range_topic', '/anchor2/tag/range')
        self.declare_parameter('anchor1_direction_topic', '/anchor1/tag/direction')
        self.declare_parameter('anchor2_direction_topic', '/anchor2/tag/direction')

        self.offset = float(self.get_parameter('anchor_offset').value)
        self.range_noise_stddev = float(
            self.get_parameter('range_noise_stddev').value)
        self.direction_noise_stddev = float(
            self.get_parameter('direction_noise_stddev').value)
        noise_seed = int(self.get_parameter('noise_seed').value)
        self.random = random.Random(noise_seed)

        if self.range_noise_stddev < 0.0:
            raise ValueError('range_noise_stddev cannot be negative')
        if self.direction_noise_stddev < 0.0:
            raise ValueError('direction_noise_stddev cannot be negative')
        self.leader_pose = None
        self.follower_pose = None

        self.r1_pub = self.create_publisher(
            Range, self._param('anchor1_range_topic'), 10)
        self.r2_pub = self.create_publisher(
            Range, self._param('anchor2_range_topic'), 10)
        self.d1_pub = self.create_publisher(
            Float32, self._param('anchor1_direction_topic'), 10)
        self.d2_pub = self.create_publisher(
            Float32, self._param('anchor2_direction_topic'), 10)

        self.create_subscription(
            Odometry, self._param('follower_odom_topic'),
            self.follower_callback, 10)
        self.create_subscription(
            Odometry, self._param('leader_odom_topic'),
            self.leader_callback, 10)

    def _param(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def follower_callback(self, msg):
        self.follower_pose = self.extract_pose(msg)
        self.update_uwb()

    def leader_callback(self, msg):
        self.leader_pose = self.extract_pose(msg)
        self.update_uwb()

    @staticmethod
    def extract_pose(msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        siny_cosp = 2.0 * (
            orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return position.x, position.y, yaw

    def update_uwb(self):
        if self.follower_pose is None or self.leader_pose is None:
            return

        fx, fy, follower_yaw = self.follower_pose
        leader_x, leader_y, _ = self.leader_pose
        lateral_yaw = follower_yaw + math.pi / 2.0
        lateral_x = self.offset * math.cos(lateral_yaw)
        lateral_y = self.offset * math.sin(lateral_yaw)

        right = (fx + lateral_x, fy + lateral_y)
        left = (fx - lateral_x, fy - lateral_y)
        range1, angle1 = self.compute_range_angle(
            *right, leader_x, leader_y, follower_yaw)
        range2, angle2 = self.compute_range_angle(
            *left, leader_x, leader_y, follower_yaw)

        range1 = max(
            0.0, range1 + self.random.gauss(0.0, self.range_noise_stddev))
        range2 = max(
            0.0, range2 + self.random.gauss(0.0, self.range_noise_stddev))
        direction1 = math.degrees(angle1) + self.random.gauss(
            0.0, self.direction_noise_stddev)
        direction2 = math.degrees(angle2) + self.random.gauss(
            0.0, self.direction_noise_stddev)

        stamp = self.get_clock().now().to_msg()
        self.r1_pub.publish(self.range_message(range1, 'anchor1', stamp))
        self.r2_pub.publish(self.range_message(range2, 'anchor2', stamp))
        self.d1_pub.publish(Float32(data=direction1))
        self.d2_pub.publish(Float32(data=direction2))

    @staticmethod
    def range_message(distance, frame_id, stamp):
        msg = Range()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = math.pi
        msg.min_range = 0.0
        msg.max_range = 100.0
        msg.range = float(distance)
        return msg

    @staticmethod
    def compute_range_angle(anchor_x, anchor_y, target_x, target_y, yaw):
        dx = target_x - anchor_x
        dy = target_y - anchor_y
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) - yaw
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return distance, angle


def main(args=None):
    rclpy.init(args=args)
    node = SimUWB()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
