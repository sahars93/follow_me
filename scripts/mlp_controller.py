#!/usr/bin/env python3
"""Run the MLP UWB follow-me policy as a ROS 2 node."""

from uwb_onnx_controller import run_controller


def main(args=None):
    run_controller('mlp_controller', 'uwb_mlp.onnx', args=args)


if __name__ == '__main__':
    main()
