import roboticstoolbox as rtb
from roboticstoolbox.backends.swift import Swift
import time

# Load robot from URDF
robot = rtb.ERobot.URDF('/home/buse/collaborice_ws/src/jaka_description/urdf/jaka_swift.urdf')

# Set a starting configuration (optional)
robot.q = [0, -1.0, 1.0, 0, 1.0, 0]  # in radians

# Launch Swift simulator
sim = Swift()
sim.launch()
sim.add(robot)

# Keep the viewer running for a while
for _ in range(100):
    sim.step(0.1)
    time.sleep(0.1)

sim.close()
