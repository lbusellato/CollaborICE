#!/usr/bin/env python3
"""
Linear extrapolation forecasting node: subclasses the template ForecastingNode and predicts future hand trajectory
by constant-velocity extrapolation using the last two frames.
"""
import argparse
import os
import json
import time
import threading
from collections import deque

import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64

from leapmotion import LeapFrame
from ament_index_python import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory

# Import the base template
try:
    from ..template_forecasting_node import ForecastingNode
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
    from template_forecasting_node import ForecastingNode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear velocity-based forecasting"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug/mockup topics"
    )
    parser.add_argument(
        "--hz", type=int, default=30,
        help="Rate in Hz for incoming data and predictions"
    )
    parser.add_argument(
        "--horizon", type=int, default=30,
        help="Number of future steps to predict"
    )
    parser.add_argument("--topic_name", type=str, default="/applications/hand_forecasting", help="Publish topic name for the node.")
    return parser.parse_known_args()


class LinearNode(ForecastingNode):
    """
    ForecastingNode subclass: assumes constant velocity from last two wrist positions.
    """
    def __init__(
        self,
        name: str = "linear_forecasting_node",
        target_hz: int = 30,
        horizon: int = 30,
        debug: bool = False,
         publish_topic_name: str|None = None,
        **kwargs
    ):
        # Initialize base with name and rate
        super().__init__(name=name, target_hz=target_hz, publish_topic_name=publish_topic_name, **kwargs)
        self.debug = debug

        # How many future frames to predict
        self.T_out = horizon
        self.seconds_in_future = self.T_out / float(self.target_hz)

        # Only need two frames to compute velocity
        self.wrist_window_deque = deque(maxlen=2)
        self.rotations_window_deque = deque(maxlen=2)

    def do_forecasting(self, input_trajectory_raw: torch.Tensor) -> dict:
        """
        Compute linear extrapolation based on velocity from last two frames.
        :param input_trajectory_raw: tensor [T_in, 3] but we only use the last two points
        :returns: dict matching output schema
        """
        # Extract last two positions
        p_prev = self.wrist_window_deque[-2]
        p_last = self.wrist_window_deque[-1]
        dt = 1.0 / float(self.target_hz)
        velocity = (p_last - p_prev) / dt

        # Build future trajectory
        forecast_positions = []
        for i in range(1, self.T_out + 1):
            t = i * dt
            forecast_positions.append((p_last + velocity * t).tolist())

        # Constant rotation (last observed)
        rot_last = self.rotations_window_deque[-1].tolist()
        forecast_rotations = [rot_last] * self.T_out

        # Package output
        return {
            "future_position": forecast_positions[-1],
            "future_trajectory": forecast_positions,
            "future_rotation": rot_last,
            "future_rotations": forecast_rotations,
            "time_seconds_in_future": self.seconds_in_future,
            "time_step_in_future": self.T_out,
            "uncertainty": {
                # linear extrapolation has no learned uncertainty
                "overall_certainty": None,
                "last_certainty": None,
                "overall_variance": None,
                "last_variance": None,
            },
        }


def main():
    # Parse args
    args, _ = parse_args()
    rclpy.init()

    # Launch node with specified parameters
    node = LinearNode(
        name="linear_forecasting_node",
        target_hz=args.hz,
        horizon=args.horizon,
        debug=args.debug,
        publish_topic_name=args.topic_name
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass 
    finally:
        node.destroy_node()
    
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()