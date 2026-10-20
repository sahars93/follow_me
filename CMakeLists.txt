cmake_minimum_required(VERSION 3.8)
project(follow_me)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/LSTM_main_uwb.py
  scripts/SVM_main_uwb.py
  scripts/gazebo_uwb_publisher.py
  scripts/leader_test_controller.py
  scripts/mlp_controller.py
  scripts/pd_controller.py
  scripts/trajectory_recorder.py
  imgs/plot_follow_me_csvs.py
  scripts/uwb_onnx_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
