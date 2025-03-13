# LeapMotion ROS2 Package

A ROS2 package for streaming data from the LeapMotion hand tracking device.

## Features

- Streams LeapMotion hand tracking data to ROS2 topics
- Provides complete hand skeletal data including all finger joints
- Includes palm position and orientation
- Includes grab and pinch metrics
- Uses standard ROS2 message types where possible
- Optimized performance with rate limiting

## Topics

- `/leap/frame` - Complete LeapMotion frame data (custom message)
- `/leap/hands/pose` - Hand palm positions and orientations (PoseArray)
- `/leap/hands/joints` - All joint positions in the hands (JointState)

## Usage

### Prerequisites

- ROS2 (tested with Humble/Iron)
- LeapMotion SDK
- LeapMotion device

### Building

```
cd ~/ros2_ws/src
git clone https://github.com/user/collaborice_leap_streamer.git
cd ..
colcon build --packages-select leap_interfaces collaborice_leap_streamer
source install/setup.bash
```

### Running

```
ros2 launch collaborice_leap_streamer leap_streamer.launch.py
```

### Launch Parameters

- `frame_id` - Frame ID for LeapMotion data (default: "leap_motion")
- `publish_rate_limit` - Maximum publish rate in Hz (default: 120.0)
- `tracking_mode` - Tracking mode, "Desktop" or "HMD" (default: "Desktop")

## License

Apache License 2.0

## Acknowledgments

This package uses the LeapMotion SDK for hand tracking data.

<!-- colcon build --packages-select collaborice_leap_mockup

source install/setup.bash

ros2 run collaborice_leap_mockup talker
 -->
