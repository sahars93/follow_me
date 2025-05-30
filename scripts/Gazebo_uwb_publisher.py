#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from std_msgs.msg import Float32
import tf

class SimUWB:
    def __init__(self):
        rospy.init_node('Gazebo_uwb_sim')

        self.offset = 0.2  # 20 cm offset to left and right

        self.leader_pose = None
        self.follower_pose = None

        # Publishers
        self.r1_pub = rospy.Publisher('/anchor1/tag/range', Range, queue_size=1)
        self.r2_pub = rospy.Publisher('/anchor2/tag/range', Range, queue_size=1)
        self.d1_pub = rospy.Publisher('/anchor1/tag/direction', Float32, queue_size=1)
        self.d2_pub = rospy.Publisher('/anchor2/tag/direction', Float32, queue_size=1)

        # Subscribers
        rospy.Subscriber('/tb3_1/odom', Odometry, self.follower_callback)
        rospy.Subscriber('/tb3_2/odom', Odometry, self.leader_callback)

    def follower_callback(self, msg):
        self.follower_pose = self.extract_pose(msg)
        self.update_uwb()

    def leader_callback(self, msg):
        self.leader_pose = self.extract_pose(msg)
        self.update_uwb()

    def extract_pose(self, odom_msg):
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2]
        return pos.x, pos.y, yaw

    def update_uwb(self):
        if self.follower_pose is None or self.leader_pose is None:
            return

        fx, fy, fyaw = self.follower_pose
        lx, ly, _ = self.leader_pose

        # Anchor positions relative to follower's body frame
        left_x = fx - self.offset * np.cos(fyaw + np.pi / 2)
        left_y = fy - self.offset * np.sin(fyaw + np.pi / 2)
        right_x = fx + self.offset * np.cos(fyaw + np.pi / 2)
        right_y = fy + self.offset * np.sin(fyaw + np.pi / 2)

        # Leader vector in each anchor's local frame
        r1, angle1 = self.compute_range_angle_local(right_x, right_y, lx, ly, fyaw)
        r2, angle2 = self.compute_range_angle_local(left_x, left_y, lx, ly, fyaw)

        # Publish range (in meters)
        range1_msg = Range()
        range1_msg.range = r1
        range1_msg.radiation_type = Range.INFRARED
        range1_msg.header.stamp = rospy.Time.now()

        range2_msg = Range()
        range2_msg.range = r2
        range2_msg.radiation_type = Range.INFRARED
        range2_msg.header.stamp = rospy.Time.now()

        # Publish direction (in radians, relative to anchor orientation)
        dir1_msg = Float32(data=np.degrees(angle1))
        dir2_msg = Float32(data=np.degrees(angle2))

        self.r1_pub.publish(range1_msg)
        self.d1_pub.publish(dir1_msg)
        self.r2_pub.publish(range2_msg)
        self.d2_pub.publish(dir2_msg)

    def compute_range_angle_local(self, anchor_x, anchor_y, target_x, target_y, anchor_yaw):
        dx = target_x - anchor_x
        dy = target_y - anchor_y
        range_val = np.hypot(dx, dy)

        # Relative angle to the leader in anchor's local frame
        angle_global = np.arctan2(dy, dx)
        angle_local = angle_global - anchor_yaw
        angle_local = (angle_local + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        return range_val, angle_local

if __name__ == '__main__':
    try:
        SimUWB()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
