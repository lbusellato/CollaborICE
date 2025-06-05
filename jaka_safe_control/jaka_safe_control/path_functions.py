import numpy as np

from scipy.spatial.transform import Slerp, Rotation as R
from tf_transformations import quaternion_from_euler

#########################################
#                                       #
# Path Functions                        #
#                                       #
#########################################

def linear_move(start_pose: list, end_pose: list, max_linear_vel: float=100):
    """Computes the nominal end-effector trajectory for a linear motion in Cartesian space towards a target maintaining constant velocity.

    Parameters
    ----------
    start_pose : list
        Starting pose for the trajectory (usually the current TCP).
    end_pose : list
        Target pose for the trajectory. 
    max_linear_vel : float, optional
        Maximum linear velocity in mm/s, by default 100

    Returns
    -------
    _type_
        _description_
    """
    start_pos = start_pose[:3]
    end_pos =   end_pose[:3]
    start_quat = np.array(quaternion_from_euler(*start_pose[3:]))
    end_quat = np.array(quaternion_from_euler(*end_pose[3:]))

    linear_displacement = np.linalg.norm(end_pos - start_pos)

    if linear_displacement < 1e-6:
        t = 1.0
    else:
        # HACK
        t = min(0.35 * max_linear_vel / linear_displacement, 1.0)


    interp_pos = start_pos + t * (end_pos - start_pos)
    interp_rot = Slerp([0, 1], R.from_quat([start_quat, end_quat]))([t]).as_euler('xyz')[0]
    return tuple(interp_pos) + tuple(interp_rot)

def triangle_wave(tau: float, 
                t: float=0.0)->tuple:
    """Computes the nominal end-effector trajectory for a fixed orientation triangle wave on the y direction.

    Parameters
    ----------
    tau : float
        Time to compute the trajectory in.
    t : float, optional
        Start time of the trajectory, by default 0.0

    Returns
    -------
    tuple
        The computed TCP pose and joint positions at time tau.
    """
    x_c, y_c, z_c = [-0.400, 0.0, 0.300]
    orientation = [np.pi, 0.0, -20*np.pi/180]
    amplitude = 0.300
    frequency = 0.1

    period = 1.0 / frequency
    phase = ((tau - t) % period) / period 
    
    x = x_c
    y = y_c + amplitude * (4 * np.abs(phase - 0.5) - 1) 
    z = z_c

    tcp_pose = np.array([x, y, z, *orientation])

    return tcp_pose