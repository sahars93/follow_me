# End-to-End UWB Follow-Me for ROS 2

`follow_me` is a ROS 2 package for UWB-based follow-me behavior. It combines a
simulated two-anchor UWB sensor with an ONNX imitation-learning controller that
publishes `geometry_msgs/msg/Twist` commands.

The default configuration expects a follower at `tb3_1` and a leader at
`tb3_2`. All robot and sensor topic names are ROS parameters and can be changed
from the launch command.

## Requirements

- ROS 2 Humble or newer
- `colcon` and `rosdep`
- Python 3 with NumPy and ONNX Runtime
- A simulator publishing `nav_msgs/msg/Odometry` for the leader and follower

Install the ROS dependencies from the root of your ROS 2 workspace:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

If your ROS distribution does not provide ONNX Runtime through `rosdep`, install
it into the same Python environment used by ROS:

```bash
python3 -m pip install onnxruntime
```

## Build

Clone the repository into a ROS 2 workspace, then build and source it:

```bash
mkdir -p ~/follow_me_ws/src
cd ~/follow_me_ws/src
git clone https://github.com/sahars93/follow_me.git follow_me_ros2
cd ~/follow_me_ws
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install --packages-select follow_me
source install/setup.bash
```

## Run in simulation

Install TurtleBot3 Gazebo and choose a TurtleBot3 model:

```bash
sudo apt install ros-$ROS_DISTRO-turtlebot3-gazebo
export TURTLEBOT3_MODEL=waffle
```

Start the TurtleBot3 empty Gazebo world:

```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

In a second terminal, spawn the follower as `tb3_1`:

```bash
export TURTLEBOT3_MODEL=waffle
ros2 run gazebo_ros spawn_entity.py \
  -entity tb3_1 \
  -file "$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models/turtlebot3_${TURTLEBOT3_MODEL}/model.sdf" \
  -robot_namespace /tb3_1 \
  -x 0.0 -y -6.0 -z 0.01 -Y 1.57
```

In a third terminal, spawn the leader as `tb3_2`:

```bash
export TURTLEBOT3_MODEL=waffle
ros2 run gazebo_ros spawn_entity.py \
  -entity tb3_2 \
  -file "$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models/turtlebot3_${TURTLEBOT3_MODEL}/model.sdf" \
  -robot_namespace /tb3_2 \
  -x 0.0 -y -4.0 -z 0.01 -Y 1.57
```

The two robots provide the default topics expected by this package:

```text
/tb3_1/odom       follower odometry
/tb3_2/odom       leader odometry
/tb3_1/cmd_vel    follower velocity command
```

In a second sourced terminal, launch the simulated UWB publisher and controller:

```bash
ros2 launch follow_me follow_me.launch.py
```

The MLP policy is selected by default. Select another installed policy with the
`controller` launch argument:

```bash
# Support-vector-machine policy
ros2 launch follow_me follow_me.launch.py controller:=svm

# 100-sample LSTM policy
ros2 launch follow_me follow_me.launch.py controller:=lstm

# Proportional-derivative baseline
ros2 launch follow_me follow_me.launch.py controller:=pd
```

Valid values are `mlp`, `svm`, `lstm`, and `pd`. The learned controllers use
the packaged models
`uwb_mlp.onnx`, `uwb_svm.onnx`, and `uwb_lstm_100_2.onnx`, respectively. The
LSTM history is sampled at `inference_rate` (50 Hz by default), giving a
2-second input window. An explicit `model_path` overrides the packaged model
for whichever learned controller is selected.

| Controller | Model input | Packaged model |
| --- | --- | --- |
| `mlp` | Current UWB sample, shape `(1, 4)` | `models/uwb_mlp.onnx` |
| `svm` | Current UWB sample, shape `(1, 4)` | `models/uwb_svm.onnx` |
| `lstm` | Last 100 UWB samples, shape `(1, 100, 4)` | `models/uwb_lstm_100_2.onnx` |
| `pd` | Current UWB distance and heading errors | No learned model |

Each sample has the order `[range1, range2, direction1, direction2]`. Ranges
are in metres. The publisher supplies directions in degrees, and the controller
converts them to radians before inference. The LSTM history is initially padded
with zeros and updated once per inference cycle.

### PD baseline

The `pd` option is a non-learning baseline using the same UWB measurements as
the learned controllers. It averages the two ranges to estimate leader distance
and uses the circular mean of the two directions to estimate leader heading.
The distance PD loop commands forward velocity, while the heading PD loop
commands angular velocity:

```text
v = kp_linear  * distance_error + kd_linear  * distance_error_rate
w = kp_angular * heading_error  + kd_angular * heading_error_rate
```

Negative linear commands are prevented so the follower does not reverse toward
the leader. Both commands are limited by `max_linear_speed` and
`max_angular_speed`. Run the baseline with:

```bash
ros2 launch follow_me follow_me.launch.py controller:=pd
```

The default PD parameters are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `desired_distance` | `1.5` | Target following distance in metres |
| `kp_linear` | `0.8` | Distance proportional gain |
| `kd_linear` | `0.15` | Distance derivative gain |
| `kp_angular` | `1.5` | Heading proportional gain |
| `kd_angular` | `0.2` | Heading derivative gain |
| `max_linear_speed` | `1.2` | Forward command limit in m/s |
| `max_angular_speed` | `1.5` | Angular command limit in rad/s |
| `distance_deadband` | `0.05` | Distance-error deadband in metres |
| `heading_deadband` | `0.03` | Heading-error deadband in radians |

For direct gain tuning, run the node separately from the UWB publisher:

```bash
ros2 run follow_me pd_controller.py --ros-args \
  -p desired_distance:=1.5 \
  -p kp_linear:=0.8 \
  -p kd_linear:=0.15 \
  -p kp_angular:=1.5 \
  -p kd_angular:=0.2
```

To use a different compatible model file:

```bash
ros2 launch follow_me follow_me.launch.py \
  controller:=lstm \
  model_path:=/absolute/path/to/model.onnx
```

Drive the leader with a ROS 2 teleoperation node, for example:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/tb3_2/cmd_vel
```

### Repeatable sudden-movement tests

For repeatable experiments, use the leader test controller instead of keyboard
teleoperation. Its default `w_shape` test drives four 8-metre straight legs at
0.6 m/s with alternating 120-degree turns at 0.5 rad/s. At each of the three
internal W vertices, the leader stops for 5 seconds before turning. The turns are
intentionally less
than 180 degrees so the leader does not reverse directly toward the follower:

```bash
ros2 run follow_me leader_test_controller.py --ros-args \
  -p scenario:=w_shape \
  -p max_linear_speed:=0.6 \
  -p publish_rate:=50.0
```

Other scenarios are `sudden_turn`, `stop_go`, `speed_step`, `slalom`, and
`combined`.
For example:

```bash
ros2 run follow_me leader_test_controller.py --ros-args \
  -p scenario:=sudden_turn
```

The node logs every transition with its commanded linear/angular velocity and
publishes `/tb3_2/cmd_vel` at 50 Hz by default. Change `publish_rate` to evaluate
another command frequency, or set `loop:=true` for a repeated endurance test.

During the W test, the same node records `/tb3_1/odom` and `/tb3_2/odom`. When
the maneuver finishes, it displays three plots comparing the leader and
follower XY trajectories, linear velocities, and angular velocities. The figure
is also saved to `/tmp/follow_me_w_results.png`. For a headless experiment:

```bash
ros2 run follow_me leader_test_controller.py --ros-args \
  -p show_plot:=false \
  -p plot_output:=/tmp/follow_me_w_results.png
```

To use different robot topics:

```bash
ros2 launch follow_me follow_me.launch.py \
  follower_odom:=/follower/odom \
  leader_odom:=/leader/odom \
  cmd_vel:=/follower/cmd_vel
```

Useful launch arguments are:

| Argument | Default | Meaning |
| --- | --- | --- |
| `controller` | `mlp` | Controller: `mlp`, `svm`, `lstm`, or `pd` |
| `follower_odom` | `/tb3_1/odom` | Follower odometry input |
| `leader_odom` | `/tb3_2/odom` | Leader odometry input |
| `cmd_vel` | `/tb3_1/cmd_vel` | Controller output |
| `model_path` | packaged model | Alternate ONNX model path |
| `inference_rate` | `50.0` | Controller frequency in Hz |
| `max_linear_speed` | `1.2` | Linear-speed limit in m/s |
| `require_leader_motion` | `true` | Wait for initial leader motion before policy starts |

The UWB simulator publishes ranges in metres and directions in degrees on:

```text
/anchor1/tag/range       sensor_msgs/msg/Range
/anchor2/tag/range       sensor_msgs/msg/Range
/anchor1/tag/direction   std_msgs/msg/Float32
/anchor2/tag/direction   std_msgs/msg/Float32
```

The simulated measurements include zero-mean Gaussian noise by default: 0.02 m
range standard deviation and 1.0 degree direction standard deviation. Override
or disable it with ROS parameters, for example:

```bash
ros2 launch follow_me follow_me.launch.py \
  range_noise_stddev:=0.0 \
  direction_noise_stddev:=0.0
```

The default `noise_seed` is `42`, making experiments reproducible.

For real hardware, run only the controller and publish the same four sensor
topics from your UWB driver. Run the desired policy directly:

```bash
# MLP
ros2 run follow_me mlp_controller.py --ros-args \
  -p cmd_vel_topic:=/robot/cmd_vel \
  -p require_leader_motion:=false

# SVM
ros2 run follow_me SVM_main_uwb.py --ros-args \
  -p cmd_vel_topic:=/robot/cmd_vel \
  -p require_leader_motion:=false

# LSTM
ros2 run follow_me LSTM_main_uwb.py --ros-args \
  -p cmd_vel_topic:=/robot/cmd_vel \
  -p require_leader_motion:=false
```

Every node topic can also be overridden with ROS parameters or standard ROS 2
remapping rules. The controller waits until all four UWB values have arrived
before producing a velocity command. By default it also waits until leader
odometry first reports nonzero linear or angular velocity. This is a one-time
start trigger: after the first detected motion, the policy stays active even if
the leader stops later. For a real UWB-only deployment without leader odometry,
disable this initial simulation gate with `require_leader_motion:=false`.

## Train a model

Training is independent of the ROS graph. Create `train/data.csv` with these
columns:

```text
r1,r2,d1,d2,v_linear,v_angular
```

The direction columns are expected in degrees. From the repository root, run:

```bash
python3 train/mlp.py
```

This trains and evaluates the network and replaces `models/uwb_mlp.onnx`.
Training requires PyTorch, pandas, scikit-learn, matplotlib, and ONNX.

## Package layout

```text
launch/follow_me.launch.py        selectable ROS 2 policy launch
scripts/gazebo_uwb_publisher.py   odometry-to-UWB simulation node
scripts/mlp_controller.py         MLP ROS 2 controller entry point
scripts/SVM_main_uwb.py           SVM ROS 2 controller entry point
scripts/LSTM_main_uwb.py          LSTM ROS 2 controller entry point
scripts/pd_controller.py          non-learning PD baseline
scripts/uwb_onnx_controller.py    shared ONNX ROS 2 controller
scripts/leader_test_controller.py repeatable leader tests and plots
models/uwb_mlp.onnx               pretrained MLP model
models/uwb_svm.onnx               pretrained SVM model
models/uwb_lstm_100_2.onnx        pretrained 100-sample LSTM model
train/mlp.py                      offline MLP training script
```

## Citation

```bibtex
@misc{salimpour2025follow,
  title={Follow-Me in Micro-Mobility with End-to-End Imitation Learning},
  author={Sahar Salimpour and Iacopo Catalano and Tomi Westerlund and
          Mohsen Falahi and Jorge Pe\~na Queralta},
  year={2025},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
