import sys
import time
import json
import math
import itertools
import numpy as np
from basic_robotics.general import fsr, tm
from basic_robotics.utilities.disp import progressBar
from moller_trumbore import inside_alpha_shape


def process_empty(p):
    """
    Helper function for multiprocessing. Literally returns empty.
    Args:
        p: Point
    Returns:
        [p, 0, [], [], []]
    """
    return [p, 0, [], [], []]


def grid_cloud_within_volume(shape, resolution=.25):
    """
    Fill an alpha shape with a grid of points according to the resolution parameter.

    Args:
        resolution: how far (in meters) points should be from each other
    Returns:
        pruned_cloud: The section of the grid space that fits within the volume
    """
    bounds_x = shape.bounds[0]
    bounds_y = shape.bounds[1]
    bounds_z = shape.bounds[2]

    def round_near(x, a):
        return math.floor(x / a) * a

    def gen_lin(bounds):
        # Generate a linearly interpolated grid between minimum and maximum
        # Bounds for the purposes of setting up a 3D grid
        min_t = -round_near(abs(bounds[0]), resolution)
        max_t = round_near(bounds[1], resolution)
        return np.arange(min_t, max_t, resolution)
        # Changed to Arange to achieve desired steps

    x_space = gen_lin(bounds_x).tolist()
    y_space = gen_lin(bounds_y).tolist()
    z_space = gen_lin(bounds_z).tolist()

    # Construct point cloud by matching every combination of the created points
    cloud_list = list(itertools.product(*[x_space, y_space, z_space]))
    cloud = np.array(cloud_list)

    # Prune Cloud by erasing points known to be
    # outside the volume of the alpha shape
    print('Initial Coud Size:' + str(len(cloud)))
    pruned_cloud = inside_alpha_shape(shape, cloud)
    print('Pruned Cloud Size:' + str(len(pruned_cloud)))
    return pruned_cloud


def ignore_close_points(seen_points, empty_results, test_point, minimum_dist):
    """
    Ignores a point too close to other close points

    Args:
        seen_points: List of already accepted points
        empty_results: Empty results list representing rejected points
        test_point: Point to check
        minimum_dist: Minimum allowable distance between points

    Returns:
        type: boolean signalling whether or not to ignore, and a list of ingored point/results

    """
    continuance = False
    # If we're ignoring the point, we do need to
    # supply an empty result to not break later analysis
    for point in seen_points:
        if fsr.distance(point, test_point) < minimum_dist:
            empty_results.append(process_empty(test_point))
            continuance = True
            break
    return continuance, empty_results


def gen_manip_sphere(manip_resolution):
    """
    Craft a manipulability sphere for use in testing manipulability resolution
    Args:
        manip_resolution: approximate number of points in sphere
    Returns:
        sphere: points in sphere
        true_rez: the true number of points in the sphere
    """
    sphr = fsr.fiboSphere(manip_resolution)
    np.vstack((np.array([0, 0, 0]), sphr))
    sphere = np.array(sphr) * np.pi
    true_rez = len(sphere)
    return sphere, true_rez


def filter_manipulability_at_threshold(results, score_threshold=0.5):
    """
    Filter Manipulability At THreshold
    (See only the parts of a workspace that have desired manipulability)

    Args:
        results: a pre-existing results list, generated from an earlier calculation
        score_threshold: desired threshold to filter above (e.g .75 = above 75% range 0-1)

    Returns:
        type: Truncated Results List
    """
    filtered_results = []
    for result in results:
        if result[1] > score_threshold:  # Should be range 0-1
            filtered_results.append(result)
    return filtered_results


def find_bounds_workspace(work_space):
    workspace_len = len(work_space)
    x_min_dim, y_min_dim, z_min_dim = np.Inf, np.Inf, np.Inf
    x_max_dim, y_max_dim, z_max_dim = -np.Inf, -np.Inf, -np.Inf
    x_min_prox, y_min_prox, z_min_prox = np.Inf, np.Inf, np.Inf

    for i in range(workspace_len):
        progressBar(i, workspace_len - 1, prefix='Finding Workspace Bounds')
        p = work_space[i][0]
        if i < workspace_len - 1:
            pn = work_space[i + 1][0]
            if abs(p[0] - pn[0]) < x_min_prox and abs(p[0] - pn[0]) > .05:
                x_min_prox = abs(p[0] - pn[0])
            if abs(p[1] - pn[1]) < y_min_prox and abs(p[1] - pn[1]) > .05:
                y_min_prox = abs(p[1] - pn[1])
            if abs(p[2] - pn[2]) < z_min_prox and abs(p[2] - pn[2]) > .05:
                z_min_prox = abs(p[2] - pn[2])
        if p[0] < x_min_dim:
            x_min_dim = p[0]
        if p[1] < y_min_dim:
            y_min_dim = p[1]
        if p[2] < z_min_dim:
            z_min_dim = p[2]
        if p[0] > x_max_dim:
            x_max_dim = p[0]
        if p[1] > y_max_dim:
            y_max_dim = p[1]
        if p[2] > z_max_dim:
            z_max_dim = p[2]

    x_min = (x_min_dim * (1 / x_min_prox))
    y_min = (y_min_dim * (1 / y_min_prox))
    z_min = (z_min_dim * (1 / z_min_prox))
    x_max = (x_max_dim * (1 / x_min_prox))
    y_max = (y_max_dim * (1 / y_min_prox))
    z_max = (z_max_dim * (1 / z_min_prox))
    print('Creating Graph Skeleton')

    if (not np.isclose(x_min_prox, z_min_prox, 0.01) and not
            np.isclose(y_min_prox, z_min_prox)):
        print('Grid is not cubic')
        print([x_min_prox, y_min_prox, z_min_prox])
        return None, None, None

    x_bound = int(np.ceil(x_max - x_min))
    y_bound = int(np.ceil(y_max - y_min))
    z_bound = int(np.ceil(z_max - z_min))

    real_bounds = [
        x_min_dim, x_max_dim, y_min_dim, y_max_dim, z_min_dim, z_max_dim
    ]

    bounds = [x_bound, y_bound, z_bound]
    return real_bounds, bounds, (x_min_prox + y_min_prox + z_min_prox)/3


def convert_to_json(results, json_file_name):
    """
    Convert a results list into a human-readable json file

    Args:
        results: Results list from workspace analyzer
        json_file_name: filename string of desired json file
    """
    json_dict = {}
    for result in results:
        id_str = str(result[0])  # Coordinate of result
        successes = result[2]
        thetas = result[3]
        json_dict[id_str] = {}
        json_dict[id_str]['score'] = result[1]
        json_dict[id_str]['successes'] = {}
        for i in range(len(successes)):
            json_dict[id_str]['successes'][str(i)] = successes[i].flatten().aslist()
        for i in range(len(thetas)):
            json_dict[id_str]['thetas'][str(i)] = thetas[i].flatten().aslist()
        with open(json_file_name, 'w') as json_file:
            json.dump(json_dict, json_file)


def convert_from_json(json_file_name):
    """
    Convert results from a json file into something useable

    Args:
        json_file_name: json file name containing results

    Returns:
        list: results as loaded from JSON
    """
    with open(json_file_name, 'r') as json_file:
        data = json.load(json_file)

    results = []
    for item in data:
        result = []
        result.append(tm(item))
        result.append(data[item]['score'])
        successes = []
        thetas = []
        for success in data[item]['successes']:
            successes.append(tm(data[item['successes'][success]]))
        for theta in data[item]['thetas']:
            thetas.append(np.array(data[item]['thetas'][theta]))
        result.append(successes)
        result.append(thetas)
        results.append(result)
    return results


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


def score_point_div(score):
    """
    Scores a point for use in plotting.

    Args:
        score: float score of point
    Returns:
        color: matplotlib color
    """
    col = 'darkviolet'
    if score > .9:
        col = 'limegreen'
    elif score > .8:
        col = 'green'
    elif score > .7:
        col = 'mediumseagreen'
    elif score < .1:
        col = 'darkred'
    elif score < .2:
        col = 'red'
    elif score < .3:
        col = 'orangered'
    elif score > .6:
        col = 'teal'
    elif score < .4:
        col = 'chocolate'
    elif score > .55:
        col = 'darkslategray'
    elif score < .45:
        col = 'saddlebrown'
    return col


def complete_trajectory(point_list, point_interpolation_dist, point_interpolation_mode):
    """
    Interpolate a given trajectory to user specifications (IK)

    Args:
        point_list: List of waypoints that comprise the trajectory
        point_interpolation_dist: maximum distance apart to place interpolated points
        point_interpolation_mode: type of interpolation to perform

    Returns:
        [tm]: list of transformation objects
    """
    if point_interpolation_dist > 0:
        new_points = []
        point_list_len = len(point_list)
        for i in range(point_list_len):
            if i == point_list_len - 1:
                # Add the last point, ensuring at least one point
                new_points.append(point_list[i])
                continue
            start_point = point_list[i]
            end_point = point_list[i + 1]
            distance = fsr.arcDistance(start_point, end_point)
            if distance < point_interpolation_dist:
                continue
            while distance > point_interpolation_dist:
                # Linear gap closes by directly interpolating between waypoints
                # and scaling by a distance, creating a linear path.
                # Arc gap creates a more curved or
                # 'arced' trajectory from waypoints A to B
                if point_interpolation_mode == 1:
                    start_point = fsr.closeLinearGap(start_point, end_point,
                                                     point_interpolation_dist)
                else:
                    start_point = fsr.closeArcGap(start_point, end_point,
                                                  point_interpolation_dist)
                distance = fsr.arcDistance(start_point, end_point)
                new_points.append(start_point)
        point_list = new_points
    return point_list


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
        if 'np.pi*' in val:
            return np.pi/float(val[6:])
        return float(val)

    with open(fname, 'r') as f_handle:
        lines = f_handle.readlines()
        tms = []
        for line in lines:
            tms.append(tm([safe_float(item.strip()) for item in line[1:-2].strip().split(',')]))
        return tms


def post_flag(flag, cmds, next_ind = 1):
    """
    Return the argument of a specified flag in a list of commands.

    Args:
        flag: string to seach for
        cmds: list of strings
        next_ind: index to increment by
    Returns:
        argument to flag
    """
    return cmds[cmds.index(flag) + next_ind]


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


def wait_for(result, message):
    """
    Wait for an asynchronous result to complete proessing and also display a waiting message

    Args:
        result: asynchronous result
        message: message to display while waiting
    """
    waiter = WaitText(message)
    while not result.ready():
        waiter.print()
        time.sleep(0.5)
    waiter.done()


class WaitText:
    """
    Simple helper class to provide feedback to user
    """

    def __init__(self, text='Processing', iter_limit=5):
        """
        Initialize WaitText
        Args:
            text: [Optional String] text to display
            iter_limit: number of '.' to print before resetting
        """
        self.text = text
        self.iter = 0
        self.iter_limit = iter_limit
        sys.stdout.write(self.text)

    def print(self):
        """
        Updates the waiting message
        """
        if self.iter == self.iter_limit:
            sys.stdout.write('\b' * self.iter_limit)
            sys.stdout.write('.')
            self.iter = 0
        else:
            sys.stdout.write('.')
        self.iter += 1

    def done(self):
        """
        Prints 'done' and advances to the next line
        """
        print(' Done')
