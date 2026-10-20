#!/usr/bin/env python3
"""Run the SVM UWB follow-me policy as a ROS 2 node."""

from uwb_onnx_controller import run_controller


def main(args=None):
    run_controller(
        'svm_controller', 'uwb_svm.onnx', absolute_linear=True, args=args)


if __name__ == '__main__':
    main()
