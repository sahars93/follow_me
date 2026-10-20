#!/usr/bin/env python3
"""Publish repeatable leader motions for follow-me evaluation."""

from dataclasses import dataclass
import math
import os

from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


@dataclass(frozen=True)
class Phase:
    """One constant-velocity section of a test maneuver."""

    name: str
    duration: float
    linear: float
    angular: float


SCENARIOS = {
    'w_shape': [
        Phase('w_leg_1', 8.0 / 0.6, 0.6, 0.0),
        Phase('w_vertex_1_stop', 5.0, 0.0, 0.0),
        Phase('w_turn_left_120', math.radians(120.0) / 0.5, 0.0, 0.5),
        Phase('w_leg_2', 8.0 / 0.6, 0.6, 0.0),
        Phase('w_vertex_2_stop', 5.0, 0.0, 0.0),
        Phase('w_turn_right_120', math.radians(120.0) / 0.5, 0.0, -0.5),
        Phase('w_leg_3', 8.0 / 0.6, 0.6, 0.0),
        Phase('w_vertex_3_stop', 5.0, 0.0, 0.0),
        Phase('w_turn_left_120_2', math.radians(120.0) / 0.5, 0.0, 0.5),
        Phase('w_leg_4', 8.0 / 0.6, 0.6, 0.0),
        Phase('stop', 2.0, 0.0, 0.0),
    ],
    'sudden_turn': [
        Phase('straight', 3.0, 0.25, 0.0),
        Phase('left_90', math.pi / 2.0, 0.0, 1.0),
        Phase('straight_after_left', 3.0, 0.25, 0.0),
        Phase('right_180', math.pi, 0.0, -1.0),
        Phase('straight_after_reversal', 3.0, 0.25, 0.0),
        Phase('stop', 2.0, 0.0, 0.0),
    ],
    'stop_go': [
        Phase('forward', 3.0, 0.30, 0.0),
        Phase('emergency_stop', 2.0, 0.0, 0.0),
        Phase('sudden_restart', 3.0, 0.45, 0.0),
        Phase('stop', 2.0, 0.0, 0.0),
    ],
    'speed_step': [
        Phase('slow', 3.0, 0.10, 0.0),
        Phase('fast_step', 3.0, 0.50, 0.0),
        Phase('slow_step', 3.0, 0.15, 0.0),
        Phase('stop', 2.0, 0.0, 0.0),
    ],
    'slalom': [
        Phase('left_arc', 2.0, 0.25, 0.8),
        Phase('right_reversal', 2.0, 0.25, -0.8),
        Phase('left_reversal', 2.0, 0.25, 0.8),
        Phase('right_reversal_2', 2.0, 0.25, -0.8),
        Phase('stop', 2.0, 0.0, 0.0),
    ],
    'zigzag_1m': [
        Phase('segment_1', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_left_90_1', math.pi / 2.0 / 0.8, 0.0, 0.8),
        Phase('segment_2', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_right_90_1', math.pi / 2.0 / 0.8, 0.0, -0.8),
        Phase('segment_3', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_left_90_2', math.pi / 2.0 / 0.8, 0.0, 0.8),
        Phase('segment_4', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_right_90_2', math.pi / 2.0 / 0.8, 0.0, -0.8),
        Phase('segment_5', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_left_90_3', math.pi / 2.0 / 0.8, 0.0, 0.8),
        Phase('segment_6', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_right_90_3', math.pi / 2.0 / 0.8, 0.0, -0.8),
        Phase('segment_7', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_left_90_4', math.pi / 2.0 / 0.8, 0.0, 0.8),
        Phase('segment_8', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_right_90_4', math.pi / 2.0 / 0.8, 0.0, -0.8),
        Phase('segment_9', 1.0 / 1.0, 1.0, 0.0),
        Phase('turn_left_90_5', math.pi / 2.0 / 0.8, 0.0, 0.8),
        Phase('segment_10', 1.0 / 1.0, 1.0, 0.0),
        Phase('stop', 2.0, 0.0, 0.0),
    ],
}
SCENARIOS['combined'] = (
    SCENARIOS['speed_step'][:-1]
    + SCENARIOS['stop_go'][:-1]
    + SCENARIOS['sudden_turn'][:-1]
    + SCENARIOS['slalom']
)


class LeaderTestController(Node):
    """Execute a selected open-loop motion sequence on the leader."""

    def __init__(self):
        super().__init__('leader_test_controller')
        self.declare_parameter('cmd_vel_topic', '/tb3_2/cmd_vel')
        self.declare_parameter('scenario', 'w_shape')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('leader_odom_topic', '/tb3_2/odom')
        self.declare_parameter('follower_odom_topic', '/tb3_1/odom')
        self.declare_parameter(
            'plot_output', '/tmp/follow_me_w_results.png')
        self.declare_parameter('show_plot', True)

        topic = self.get_parameter('cmd_vel_topic').value
        scenario_name = self.get_parameter('scenario').value
        rate = float(self.get_parameter('publish_rate').value)
        self.max_linear_speed = float(
            self.get_parameter('max_linear_speed').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.plot_output = self.get_parameter('plot_output').value
        self.show_plot = bool(self.get_parameter('show_plot').value)

        if scenario_name not in SCENARIOS:
            choices = ', '.join(sorted(SCENARIOS))
            raise ValueError(
                f'Unknown scenario {scenario_name!r}; choose from: {choices}')
        if rate <= 0.0:
            raise ValueError('publish_rate must be greater than zero')
        if self.max_linear_speed <= 0.0:
            raise ValueError('max_linear_speed must be greater than zero')

        self.scenario_name = scenario_name
        self.phases = SCENARIOS[scenario_name]
        self.publisher = self.create_publisher(Twist, topic, 10)
        self.recording_start = self.get_clock().now()
        self.leader_data = self.empty_record()
        self.follower_data = self.empty_record()
        self._leader_odom_subscription = self.create_subscription(
            Odometry,
            self.get_parameter('leader_odom_topic').value,
            lambda msg: self.record_odometry(msg, self.leader_data),
            10,
        )
        self._follower_odom_subscription = self.create_subscription(
            Odometry,
            self.get_parameter('follower_odom_topic').value,
            lambda msg: self.record_odometry(msg, self.follower_data),
            10,
        )
        self.phase_index = 0
        self.phase_start = self.get_clock().now()
        self.finished = False
        self.results_plotted = False
        self.timer = self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f'Starting {scenario_name!r} on {topic} at {rate:.1f} Hz')
        self.log_phase()

    def log_phase(self):
        phase = self.phases[self.phase_index]
        self.get_logger().info(
            f'Phase {self.phase_index + 1}/{len(self.phases)}: '
            f'{phase.name}, duration={phase.duration:.2f} s, '
            f'linear={phase.linear:.2f} m/s, '
            f'angular={phase.angular:.2f} rad/s')

    @staticmethod
    def empty_record():
        return {'time': [], 'x': [], 'y': [], 'linear': [], 'angular': []}

    def record_odometry(self, msg, record):
        """Record planar pose and measured velocity from an odometry message."""
        elapsed = (
            self.get_clock().now() - self.recording_start).nanoseconds / 1e9
        linear = msg.twist.twist.linear
        record['time'].append(elapsed)
        record['x'].append(msg.pose.pose.position.x)
        record['y'].append(msg.pose.pose.position.y)
        record['linear'].append(math.hypot(linear.x, linear.y))
        record['angular'].append(msg.twist.twist.angular.z)

    def plot_results(self):
        """Save and optionally display trajectory and velocity comparisons."""
        if not self.leader_data['time'] or not self.follower_data['time']:
            self.get_logger().warning(
                'Cannot plot results because one or both odometry streams are empty')
            return

        figure, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(
            self.leader_data['x'], self.leader_data['y'], label='Leader')
        axes[0].plot(
            self.follower_data['x'], self.follower_data['y'], label='Follower')
        axes[0].set_title('Robot trajectories')
        axes[0].set_xlabel('x [m]')
        axes[0].set_ylabel('y [m]')
        axes[0].axis('equal')

        axes[1].plot(
            self.leader_data['time'], self.leader_data['linear'], label='Leader')
        axes[1].plot(
            self.follower_data['time'], self.follower_data['linear'],
            label='Follower')
        axes[1].set_title('Linear velocity')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Speed [m/s]')

        axes[2].plot(
            self.leader_data['time'], self.leader_data['angular'], label='Leader')
        axes[2].plot(
            self.follower_data['time'], self.follower_data['angular'],
            label='Follower')
        axes[2].set_title('Angular velocity')
        axes[2].set_xlabel('Time [s]')
        axes[2].set_ylabel('Angular velocity [rad/s]')

        for axis in axes:
            axis.grid(True)
            axis.legend()
        figure.tight_layout()

        output_directory = os.path.dirname(os.path.abspath(self.plot_output))
        os.makedirs(output_directory, exist_ok=True)
        figure.savefig(self.plot_output, dpi=200, bbox_inches='tight')
        self.get_logger().info(f'Saved experiment plots to {self.plot_output}')
        if self.show_plot:
            plt.show()
        else:
            plt.close(figure)

    def tick(self):
        if self.finished:
            self.publisher.publish(Twist())
            if not self.results_plotted:
                self.results_plotted = True
                self.plot_results()
            return

        elapsed = (self.get_clock().now() - self.phase_start).nanoseconds / 1e9
        phase = self.phases[self.phase_index]
        if elapsed >= phase.duration:
            self.phase_index += 1
            if self.phase_index >= len(self.phases):
                if self.loop:
                    self.phase_index = 0
                else:
                    self.finished = True
                    self.publisher.publish(Twist())
                    self.get_logger().info(
                        f'Scenario {self.scenario_name!r} complete; leader stopped')
                    return
            self.phase_start = self.get_clock().now()
            phase = self.phases[self.phase_index]
            self.log_phase()

        command = Twist()
        command.linear.x = max(
            -self.max_linear_speed,
            min(self.max_linear_speed, phase.linear),
        )
        command.angular.z = phase.angular
        self.publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = LeaderTestController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publisher.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
