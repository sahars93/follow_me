# follow_me

**ROS1 package for Follow-Me behavior using UWB for End-to-End Imitation Learning.**

---

## Installation and Setup

Clone this repository into your ROS workspace and build it:

```bash
cd ~/catkin_ws/src
git clone https://github.com/sahars93/follow_me.git
cd ..
catkin_make
source devel/setup.bash
```

---

## Training

Once you have recorded your demonstration data — including UWB ranges and angles — you can train the model using:

```bash
python3 train/train_mlp.py
```

The trained model will be saved in ONNX format inside the `models/` directory.

We have provided a pretrained model for your convenience and testing.

---

## Running the Follow-Me Node

To test the model in a ROS environment, launch the node using:

```bash
roslaunch follow_me follow_me.launch
```

> **Note:**  
> - Replace the `Gazebo_uwb_publisher` node with your own UWB data publisher as needed.  
> - Make sure to update the topic names in the subscribers and publishers within the node to match your specific data topics and velocity command topics.

---

## Testing in Gazebo Simulation

1. **Install TurtleBot3 Gazebo package:**

```bash
sudo apt-get install ros-noetic-turtlebot3-gazebo
```

2. **Launch the multi-robot simulation world:**

```bash
roslaunch turtlebot3_gazebo multi_turtlebot3.launch \
second_tb3_x_pos:=0.0 second_tb3_y_pos:=-6.0 \
third_tb3_x_pos:=0.0 third_tb3_y_pos:=-4.0 third_tb3_yaw:=1.57
```

3. **Run the follow_me ROS node:**

```bash
roslaunch follow_me follow_me.launch
```

4. **Control the leader TurtleBot manually:**

```bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch cmd_vel:=/tb3_2/cmd_vel

```

This allows you to drive the leader robot, while the follower robot uses the UWB-based model to follow.

---