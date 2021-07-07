from faser_math import tm
import numpy as np
import pickle

def score_point(score):
    """
    Scores a point for use in plotting.

    Args:
        score: float score of point
    Returns:
        color: matplotlib color
    """
    col = 'darkred'
    if score > .9:
        col = 'limegreen'
    elif score > .8:
        col = 'green'
    elif score > .7:
        col = 'teal'
    elif score > .6:
        col = 'dodgerblue'
    elif score > .5:
        col = 'blue'
    elif score > .4:
        col = 'yellow'
    elif score > .3:
        col = 'orange'
    elif score > .2:
        col = 'peru'
    elif score > .1:
        col = 'red'
    return col

def load_point_cloud_from_file(fname):
    """
    Load a point cloud from a file.

    Args:
        fname: file to load from
    Returns:
        point cloud (tm)
    """
    def safe_float(val):
        if 'np.pi/' in val:
            return np.pi/float(val[6:])
        elif 'np.pi*' in val:
            return np.pi/float(val[6:])
        else:
            return float(val)

    with open(fname, 'r') as f_handle:
        lines = f_handle.readlines()
        tms = []
        for line in lines:
            tms.append(tm([safe_float(item.strip()) for item in line[1:-2].strip().split(',')]))
        return tms

def post_flag(flag, cmds):
    """
    Return the argument of a specified flag in a list of commands.

    Args:
        flag: string to seach for
        cmds: list of strings
    Returns:
        argument to flag
    """
    return cmds[cmds.index(flag) + 1]


def sort_cloud(cloud):
    """
    Sort a cloud of points to only have unique points.

    Args:
        cloud: cloud of points
    Returns:
        sorted points
    """
    list_array_input = []
    for point in cloud:
        p = point.TAA.flatten()
        list_array_input.append([p[0], p[1], p[2]])
    array_to_process = np.array(list_array_input)
    processed_array = np.unique(array_to_process.round(decimals=1), axis=0)
    return processed_array
