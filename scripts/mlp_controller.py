#!/usr/bin/env python3

import rospkg
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
import numpy as np
import onnx
import onnxruntime as ort
from collections import deque
import os

class ONNXROSInference:
    def __init__(self, model_path):
        # Initialize the ROS node
        rospy.init_node('mlp_controller_node', anonymous=True)

        self.latest_data = [0.0, 0.0, 0.0, 0.0]
        self.max_speed = 1.2

        # Load and validate the ONNX model
        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)
        rospy.loginfo("ONNX model is valid")

        # Set up the ONNX runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Subscribe to ROS topics
        rospy.Subscriber('/anchor1/tag/range', Range, self.anchor1_range_callback)
        rospy.Subscriber('/anchor2/tag/range', Range, self.anchor2_range_callback)
        rospy.Subscriber('/anchor1/tag/direction', Float32, self.anchor1_direction_callback)
        rospy.Subscriber('/anchor2/tag/direction', Float32, self.anchor2_direction_callback)

        # Set up the publisher for cmd_vel
        self.cmd_vel_publisher = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=50)

    def anchor1_range_callback(self, msg):
        self.update_latest_data(index=0, value=msg.range)

    def anchor2_range_callback(self, msg):
        self.update_latest_data(index=1, value=msg.range)

    def anchor1_direction_callback(self, msg):
        self.update_latest_data(index=2, value=msg.data)

    def anchor2_direction_callback(self, msg):
        self.update_latest_data(index=3, value=msg.data)

    def update_latest_data(self, index, value):

        self.latest_data[index] = value
        
        

    def preprocess_input(self):
        # Convert deque to NumPy array
        input_data = np.array(self.latest_data, dtype=np.float32)

        # Convert directions to radians
        input_data[2:4] = np.radians(input_data[2:4])

        return input_data.reshape(1, 4)

    def run_inference(self):
        # Prepare the input data
        input_data = self.preprocess_input()
        print(input_data)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        # Extract velocity commands from the output
        linear_velocity = outputs[0][0][0]  # Assuming the first output is linear velocity
        angular_velocity = outputs[0][0][1]  # Assuming the second output is angular velocity
        linear_velocity = max(0.0, min(self.max_speed, linear_velocity))

        # Publish the cmd_vel command
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_velocity
        cmd_vel_msg.angular.z = angular_velocity
        self.cmd_vel_publisher.publish(cmd_vel_msg)

        rospy.loginfo(f"Published cmd_vel: linear.x={linear_velocity}, angular.z={angular_velocity}")

    def spin(self):
        rate = rospy.Rate(50)  # 10 Hz
        while not rospy.is_shutdown():
            self.run_inference()
            rate.sleep()

if __name__ == "__main__":
    try:
        # Provide the path to your ONNX model
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('follow_me')
        model_path = os.path.join(pkg_path, 'models', 'uwb_mlp.onnx')
        inference_node = ONNXROSInference(model_path)
        inference_node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down ONNX ROS Inference node.")
