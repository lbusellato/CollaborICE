a = [-0.06282408535480499, 0.210161030292511, 0.07677855342626572]

import numpy as np

def leap_to_jaka(leap_pose: list,
                 R: np.ndarray= np.array([[0,-1,0], 
                                          [0,0,1], 
                                          [-1,0,0]]),
                 t: np.ndarray= np.array([0.50, -0.375, 0.01]))->np.ndarray:
    return (((np.array(leap_pose[:3])) @ R) - t) * 1000

print(leap_to_jaka(a))