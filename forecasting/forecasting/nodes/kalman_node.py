#!/usr/bin/env python3
"""
Kalman forecasting node: subclasses ForecastingNode and predicts future hand trajectory
using a linear Kalman filter over past observations.
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from leapmotion import LeapFrame

# Import the base template node
try:
    from ..template_forecasting_node import ForecastingNode
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
    from template_forecasting_node import ForecastingNode


def parse_args():
    parser = argparse.ArgumentParser(description="Kalman filter based forecasting node")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug/mockup mode")
    parser.add_argument("--hz", type=int, default=30, help="Input data rate (Hz)")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in number of steps")
    parser.add_argument("--topic_name", type=str, default="/applications/hand_forecasting", help="Publish topic name for the node.")
    return parser.parse_known_args()


class KalmanFilter:
    """
    A simple linear Kalman Filter for 3D position+velocity.
    State vector: [x, y, z, vx, vy, vz].
    """

    def __init__(self, dt, Q=None, R=None):
        # Time step
        self.dt = dt
        # State-transition matrix
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt
        # Observation matrix: we observe positions only
        self.H = np.zeros((3, 6))
        self.H[:, :3] = np.eye(3)
        # Covariance matrices
        self.Q = Q if Q is not None else np.eye(6) * 1e-4
        self.R = R if R is not None else np.eye(3) * 1e-2
        # Initialize state and covariance
        self.x = np.zeros(6)
        self.P = np.eye(6)

    def initialize(self, first_pos, first_vel=None):
        """
        Set initial state with position and optional velocity.
        """
        self.x[:3] = first_pos
        if first_vel is not None:
            self.x[3:] = first_vel
        self.P = np.eye(6)

    def update(self, z):
        """
        Perform one predict-update cycle with measurement z (3-vector).
        """
        # Predict
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        # Update
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(6) - K.dot(self.H)).dot(self.P)

    def predict(self, steps):
        """
        Forecast future positions for `steps` timesteps ahead.
        Returns lists of positions and variances.
        """
        traj = []
        covs = []
        x_pred = self.x.copy()
        P_pred = self.P.copy()
        for _ in range(steps):
            x_pred = self.F.dot(x_pred)
            P_pred = self.F.dot(P_pred).dot(self.F.T) + self.Q
            traj.append(x_pred[:3].tolist())
            covs.append(np.diag(P_pred)[:3].tolist())
        return traj, covs


class KalmanNode(ForecastingNode):
    """
    ForecastingNode subclass using a Kalman Filter.
    """

    def __init__(self, name: str = "kalman_forecasting_node", target_hz: int = 30, debug: bool = False,  publish_topic_name: str|None = None, **kwargs):
        super().__init__(name=name, target_hz=target_hz, publish_topic_name=publish_topic_name, **kwargs)
        self.debug = debug
        # Forecast horizon
        self.T_out = 30  # set by template or launch
        # Override deques based on template T_in
        self.T_in = getattr(self, "window_size", None)
        if self.T_in is None:
            # If template doesn't set window_size, default to horizon
            self.T_in = target_hz
        # Recreate sliding windows
        self.wrist_window_deque = deque(maxlen=self.T_in)
        self.rotations_window_deque = deque(maxlen=self.T_in)
        # Compute seconds_in_future
        self.seconds_in_future = getattr(self, "seconds_in_future", self.T_out * (1.0 / target_hz))

    def do_forecasting(self, input_trajectory_raw):
        # Convert to numpy array
        meas = input_trajectory_raw.cpu().numpy()
        dt = 1.0 / float(self.target_hz)
        # Initial velocity estimate
        if meas.shape[0] >= 2:
            vel0 = (meas[1] - meas[0]) / dt
        else:
            vel0 = np.zeros(3)
        # Instantiate filter
        kf = KalmanFilter(dt)
        kf.initialize(meas[0], vel0)
        # Feed measurements
        for z in meas:
            kf.update(z)
        # Forecast
        steps = self.T_out or len(meas)
        traj, covs = kf.predict(steps)
        # Constant last rotation
        last_rot = self.rotations_window_deque[-1].tolist()
        rots = [last_rot] * steps
        return {
            "future_position": traj[-1],
            "future_trajectory": traj,
            "future_rotation": last_rot,
            "future_rotations": rots,
            "time_seconds_in_future": self.seconds_in_future,
            "time_step_in_future": steps,
            "uncertainty": {
                "overall_variance": float(np.mean(covs)),
                "last_variance": covs[-1],
                "overall_certainty": None,
                "last_certainty": None,
            },
        }


def main():
    args, _ = parse_args()
    rclpy.init()
    node = KalmanNode(name="kalman_forecasting_node", target_hz=args.hz, debug=args.debug, publish_topic_name=args.topic_name)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
