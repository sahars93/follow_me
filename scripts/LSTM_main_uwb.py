#!/usr/bin/env python3
"""Run the 100-sample LSTM UWB follow-me policy as a ROS 2 node."""

from uwb_onnx_controller import run_controller


def main(args=None):
    run_controller('lstm_controller', 'uwb_lstm_100_2.onnx', args=args)


if __name__ == '__main__':
    main()
