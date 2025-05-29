#!/usr/bin/env python3
import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory

def find_latest_log(log_dir):
    """Return full path of the newest .csv in log_dir (lexicographically)."""
    files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No .csv files found in {log_dir}")
    files.sort()
    return os.path.join(log_dir, files[-1])

def load_log(filepath):
    """Read header and data (floats) from a CSV file."""
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = np.array([list(map(float, row)) for row in reader])
    return header, data

def plot_data(header, data):
    t = data[:, header.index('time')]

    # End-effector
    ee_idxs = [header.index(k) for k in ('ee_x','ee_y','ee_z')]
    ee = data[:, ee_idxs]

    # Joint commands
    u_nom_idxs  = [i for i,h in enumerate(header) if h.startswith('u_nom_')]
    u_safe_idxs = [i for i,h in enumerate(header) if h.startswith('u_safe_')]
    u_nom  = data[:, u_nom_idxs]
    u_safe = data[:, u_safe_idxs]

    # Safety functions
    h     = data[:, header.index('h')]
    has_hstar = 'hstar' in header
    if has_hstar:
        hstar = data[:, header.index('hstar')]

    # 1) EE pos
    plt.figure()
    for i,ax in enumerate(('x','y','z')):
        plt.plot(t, ee[:,i], label=ax)
    plt.title('End-Effector Position vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.grid()
    plt.legend()

    # 2) Joint vel
    plt.figure()
    for j in range(u_nom.shape[1]):
        plt.plot(t, u_nom[:,j],  '--', label=f'u_nom[{j}]')
        plt.plot(t, u_safe[:,j], '-',  label=f'u_safe[{j}]')
    plt.title('Joint Velocities: Nominal (--) vs Safe (-)')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint vel [rad/s]')
    plt.grid()
    plt.legend(ncol=2, loc='upper right')

    # 3) Safety function
    plt.figure()
    plt.plot(t, h,     label='h')
    plt.plot(t, [0]*len(t),     label='h=0')
    if has_hstar:
        plt.plot(t, hstar, label='hstar')
    plt.title('Safety Function vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('h value')
    plt.grid()
    plt.legend()

    plt.show()

def main():
    p = argparse.ArgumentParser(
        description="Plot the latest (or specified) JAKA CBF log CSV"
    )
    p.add_argument('-f','--file', help="Path to log CSV (overrides auto-search)")
    args = p.parse_args()

    if args.file:
        log_file = args.file
    else:
        share = get_package_share_directory('jaka_safe_control')
        log_dir = os.path.join(share, 'logs')
        log_file = find_latest_log(log_dir)

    print(f"Loading log: {log_file}")
    header, data = load_log(log_file)
    plot_data(header, data)

if __name__ == '__main__':
    main()
