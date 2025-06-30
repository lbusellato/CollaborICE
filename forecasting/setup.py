from glob import glob
import os
from setuptools import find_packages, setup

package_name = "forecasting"

checkpoint_files = glob(f"checkpoints/*.ckpt") + glob(f"checkpoints/*.pth")
if len(checkpoint_files) == 0:
    print("[WARNING] no checkpoint file found for forecasting node. Unexpected, this will cause the node to crash.")

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/checkpoints", checkpoint_files),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fcunico",
    maintainer_email="federico.cunico@univr.it",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # "leap_forecasting = forecasting.forecasting:main",  # This line is the one that makes the node executable
            'nn_node = forecasting.nodes.nn_node:main',
            'linear_node = forecasting.nodes.linear_node:main',
            'kalman_node = forecasting.nodes.kalman_node:main',
            'particle_node = forecasting.nodes.particle_node:main',        ],
    },
)
