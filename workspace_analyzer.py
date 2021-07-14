import math
import sys
import numpy as np
import time
from datetime import datetime
from alpha_shape import AlphaShape
from workspace_helper_functions import WaitText
from stl import mesh
import scipy as sci
# from faser_plotting.Draw.Draw import *
import multiprocessing as mp
import itertools
sys.path.append('../')
from faser_utils.disp.disp import disp, progressBar
from faser_math import tm, fsr


#Constant Values
EPSILON = .000001  # Deviation Acceptable for Moller Trumbore
RAY_UNIT = 10  # Unit Vector Constant in Moller Trumbore. Arbitrary
MAX_DIST = 10  # Maximum distance from robot base to discount points in object surface
UNIQUE_DECIMALS = 2  # Decimal places to filter out for uniqueness in Alpha Shapes

# Number of DOF from the End Effector assumed to be capable of creating a 3d shape
# Through varying of the last n DOF and tracking End Effector coordinates.
# Insufficient DOF OFFSET can result in the alpha shape reachability solver failing.
DOF_OFFSET = 3

#The ALPHA_VALUE is the alpha parameter  which determines which points are included or
# excluded to create an alpha shape. This is technically an optional parameter,
# however when the alpha shape optimizes this itself, for large clouds of points
# it can take many hours to reach a solution. 1.2 is a convenient constant tested on a variety
# of robot sizes.
ALPHA_VALUE = 1.2

#Constant values DOF_OFFSET, ALPHA_VALUE, UNIQUE_DECIMALS, and MAX_DIST are used to initialize
#Class variables of the same function, to allow tuning *if required*
#They are reproduced here at the top of the file as constants for visibility and convenience


# Helper Functions Tha You Probably Shouldn't Need To Call Directly
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
        min_t = -round_near(abs(bounds[0]), resolution)
        max_t = round_near(bounds[1], resolution)
        step = (max_t - min_t) / resolution
        return np.linspace(min_t, max_t, step)

    x_space = gen_lin(bounds_x).tolist()
    y_space = gen_lin(bounds_y).tolist()
    z_space = gen_lin(bounds_z).tolist()

    #Construct point cloud
    #cloud_list = [[]]
    cloud_list = list(itertools.product(*[x_space, y_space, z_space]))

    cloud = np.array(cloud_list)

    #Prune Cloud
    print('Initial Coud Size:' + str(len(cloud)))
    pruned_cloud = inside_alpha_shape(shape, cloud)
    print('Pruned Cloud Size:' + str(len(pruned_cloud)))
    return pruned_cloud


def moller_trumbore_ray_intersection(origin_point,
                                     triangle,
                                     ray_dir=[0, 0, 1]):
    """
    Execute Moller-Trumbore triangle/Ray Intersection Algorithm on a single point
    Args:
        origin_point: the point at which the ray originates
        triangle: List of three points in 3D space denoting a triangle
        ray_dir = Optional List of three points denoting a unit vector/ray
    Returns:
        Boolean for whether or not the ray intersects the triangle
    """
    # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    vec_0 = triangle[0]
    vec_1 = triangle[1]
    vec_2 = triangle[2]
    ray = np.array(ray_dir)
    edge_1 = vec_1 - vec_0
    edge_2 = vec_2 - vec_0
    h = np.cross(ray, edge_2)
    a = np.dot(edge_1, h)
    # if (a > -EPSILON and a < EPSILON):
    if -EPSILON < a < EPSILON:
        return False  # Ray parallel to triangle
    f = 1.0 / a
    s = origin_point - vec_0
    u = f * (np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge_1)
    v = f * np.dot(ray, q)
    if v < 0.0 or (u + v) > 1.0:
        return False
    t = f * np.dot(edge_2, q)
    if t > EPSILON:
        return True
    return False


def moller_trumbore_ray_intersection_array(points,
                                           triangle,
                                           ray=np.array([0, 0, 1])):
    """
    Executes Moller-Trumbore triangle/Ray Intersection Algorithm on a group
    of points simultaneously, utilizing numpy array expressions.
    Args:
        points: A list of points to test on the algorithm
        triangle: List of three points in 3D space denoting a triangle
        ray = Optional array of three points denoting a unit vector/ray
    Returns:
        Numpy array of booleans for which corresponding points intersect the
            triangle
    """
    vec_0 = triangle[0]
    vec_1 = triangle[1]
    vec_2 = triangle[2]
    alen = len(points)
    #print("Length: " + str(alen))
    if alen == 3:
        return moller_trumbore_ray_intersection(points, triangle)

    edge_1 = vec_1 - vec_0
    edge_2 = vec_2 - vec_0

    h = np.cross(ray, edge_2)
    a = np.dot(edge_1, h)

    if -EPSILON < a < EPSILON:
        return np.array([False] * alen)  # Ray is parallel to triangle

    # Here it becomes Matrix Ops
    f = 1.0 / a

    res = np.array([True] * alen)
    s = points - vec_0

    u = f * np.dot(s, h)

    #print(len(u))
    res[np.where(u < 0.0)] = False
    res[np.where(u > 1.0)] = False

    q = np.cross(s, edge_1)
    v = f * (np.dot(q, ray))
    res[np.where(v < 0.0)] = False
    res[np.where((u + v) > 1.0)] = False
    t = f * np.dot(q, edge_2)

    res[np.where(t <= EPSILON)] = False

    return res

def chunk_moller_trumbore(test_points, triangles, epsilon_vector,
                          result_array):
    """
    Perform a chunk of moller_trumbore, useful for parallel processing.

    Args:
        test_points: points to process [nx3]
        triangles: list/array of triangles to test with moller_trumbore
        epsilon_vector: epsilon used for Moller Trumbore
        result_array: [nx1] vector of ints used to count triangle intersections
    Returns:
        result_array
    """
    for i in range(np.shape(triangles)[0]):
        res = moller_trumbore_ray_intersection_array(test_points,
                                                     triangles[i, :],
                                                     epsilon_vector)
        # where res == True
        result_array[np.where(res)] += 1
    return result_array

def inside_alpha_shape(shape, points, check_bounds=True):
    """
    This problem is a little more complicated than it seems on its face, and requires
    a decent amount of computation, depending on the complexity of the alpha shape.
    Idealized Proceedure:
        1. Utilize an RTree or Min/Max bounds to determine if a point lies within
            the encapsulating rectangle of the alpha shape. Eliminate those that
            Do not.
        2. For those within the bounding rectangle, utilize the point-in-polygon
            algorithm applied to triangular vertices of alpha shape.*
            A) For any given point, create a vector to another point outside the
                bounding rectangle.
            B) If this ray passes through an even number of polygons, the original
                point is outside of the alpha shape.
            C) If this ray passes through an odd number of polygons, it is inside
                the alpha shape.
        3. Repeat (2) for each point in the points list that has not been pruned by
            The bounding rectangle.
        *Moller-Trumbore Ray Intersection Algorithm willl be used for this task due
        to speed constraints, as location of intersectioin is irrelevant, only that
        an intersection occurs.
        https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        https://en.wikipedia.org/wiki/Point_in_polygon
    Args:
        shape: AlphaShape object representing bounds
        points: nx3 numpy array of points to be checked
        check_bounds: [Optional Boolean] enable to trim points out of bounds before
            executing moller-trumbore
    Returns:
        list of points that are inside the AlphaShape
    """
    bounds = shape.bounds

    def isodd(x):
        return x % 2 != 0

    inside_alpha = []
    if check_bounds:
        num_points = len(points)
        i = 0
        for point in points:
            progressBar(i, num_points - 1, prefix='Checking Bounds             ')
            i += 1
            if (point[0] < bounds[0][0] or point[0] > bounds[0][1]
                    or point[1] < bounds[1][0] or point[1] > bounds[1][1]
                    or point[2] < bounds[2][0] or point[2] > bounds[2][1]):
                continue
            else:
                inside_alpha.append(point)
        inside_alpha = np.array(inside_alpha)
    else:
        inside_alpha = points
    if len(inside_alpha) == 0:
        return []
    triangles = shape.triangles
    points = inside_alpha
    num_tri = len(triangles)
    inside = []

    i = 0
    num_intersections = np.zeros(len(points))
    for tri in triangles:
        progressBar(i, num_tri - 1, prefix='Running Moller Trumbore     ')
        i += 1
        res = moller_trumbore_ray_intersection_array(
            points, tri, np.array([EPSILON, 0, RAY_UNIT - EPSILON]))
        # where res == true
        num_intersections[np.where(res)] += 1

    if len(np.where(isodd(num_intersections))[0]) == 0:
        return []
    inside = points[np.where(isodd(num_intersections))]
    return inside

def alpha_intersection(shape_1, shape_2):
    """
    Return a cut down version of shape_1, including only what is in shape_2
    Args:
        shape_1: Alpha shape to be trimmed
        shape_2: Alpha shape to specify trim
    Returns:
        new_shape: Shape created via the existing boundaries
    """
    inside_shape_1 = inside_alpha_shape(shape_1, shape_2.triangles)
    inside_shape_2 = inside_alpha_shape(shape_2, shape_1.triangles)
    new_points = np.concatenate(inside_shape_1, inside_shape_2)
    new_shape = AlphaShape(new_points, ALPHA_VALUE)
    return new_shape

def optimize_robot_for_goals(build_robot, goals_list, init=None):
    """
    Design a robot given a task space.

    Requires using Faser Kinematics Module. Can't plan around others
    Args:
        build_robot: function for building a robot. Must be FASER Incompatible
        goals_list: list of transformations that the robot must be able to reach
        init: initial link length configurations of the robot
    Returns:
        Ideal link length configuration
    """
    bot = build_robot(None)
    num_dofs = len(bot.link_home_positions) + 1
    x_init = np.ones(num_dofs)
    x_init[-2:] = .5
    if init is not None:
        x_init = init

    def lambda_helper(x, build_robot=build_robot, goals_list=goals_list):
        failures = 0
        bot = build_robot(x)
        for goal in goals_list:
            _, suc = bot.IK(goal)
            if not suc:
                failures += 1

        objective_value = sum(abs(x))**failures  #Pending Left Tab 3 (1/.5)

        print([objective_value, failures, np.round(x, 2)])
        return objective_value

    res = sci.optimize.fmin(lambda_helper, x_init)

    disp(res)
    return res

def maximize_manipulability_at_point(bot, pos):
    """
    Attemps to find the rotational configuration at a point that yields
    maximal manipulability
    Args:
        bot: robot handle
        pos: position to try (3dof)
    Returns:
        manipulability_score (float)
        max_manip_pos (transform)
    """
    def maximize_helper(pos, bot=bot):
        theta, suc = bot.IK(pos)
        if not suc:
            return 0
        mrv, mrw = calculate_manipulability_score(bot, theta)
        return (mrv + mrw) / 2

    def lambda_handle(x, pos=pos):
        return 1 - maximize_helper(tm([pos[0], pos[1], pos[2], x[0], x[1], x[2]]))

    rot_init = np.zeros(3)
    sol = sci.optimize.fmin(lambda_handle, rot_init, disp=False)
    max_manip_pos = tm([pos[0], pos[1], pos[2], sol[0], sol[1], sol[2]])
    theta, suc = bot.IK(max_manip_pos)
    if not suc:
        return 0, tm()
    score = sum(calculate_manipulability_score(bot, theta)) / 2
    return score, max_manip_pos


def calculate_manipulability_score(bot, theta):
    """
    Calculates the manipulability ellipsoid at a given pose

    Returns a 'score' of the total manipulability
    Args:
        theta: pose to test
    Returns
        float: manipulability score 0-1
    """
    jacobian_body = bot.jacobianBody(theta)
    jw = jacobian_body[0:3, :]
    jv = jacobian_body[3:6, :]

    a_w = jw @ jw.T
    a_v = jv @ jv.T

    aw_eig, _ = np.linalg.eig(a_w)
    av_eig, _ = np.linalg.eig(a_v)

    manipulability_ratio_w = 1 / (np.sqrt(max(aw_eig)) / np.sqrt(min(aw_eig)))
    #disp(manipulability_ratio_w)
    manipulability_ratio_v = 1 / (np.sqrt(max(av_eig)) / np.sqrt(min(av_eig)))
    #disp(manipulability_ratio_v)
    return manipulability_ratio_v, manipulability_ratio_w


def process_empty(p, store_joint_locations=False):
    """
    Helper function for multiprocessing. Literally returns empty
    Args:
        p: Point
        store_joint_locations: [Optional Boolean]
    Returns:
        [p, 0, []]
    """
    if store_joint_locations:
        return [p, 0, [], []]
    return [p, 0, []]


def process_point(p,
                  sphere,
                  true_rez,
                  bot,
                  store_joint_locations=False,
                  use_jacobian=False):
    """
    Process a point against a unit sphere for success metric.

    Args:
        p: Point to process
        sphere: unit sphere
        true_rez: the true amount of points inside the unit sphere
        bot: reference to the robot object
        store_joint_locations: [Optional Boolean] whether or not to store joint lcoations
        use_jacobian
    Returns:
        Results: [point, success_percentage, successful_orientations]
    """
    successes = []
    success_count = 0
    if not store_joint_locations and use_jacobian:
        score, max_pos = maximize_manipulability_at_point(bot, p)
        if score == 0:
            return [p, 0, []]
        return [p, score, [max_pos]]
    elif store_joint_locations:
        jlocs = []
    for j in range(true_rez):
        rot = sphere[j, :]
        goal = tm([p[0], p[1], p[2], rot[0], rot[1], rot[2]])
        _, suc = bot.IK(goal)
        if suc:
            successes.append(rot)
            if store_joint_locations:
                jlocs.append(bot.getJointTransforms())
            success_count += 1
        elif store_joint_locations:
            jlocs.append(0)
    if store_joint_locations:
        return [p, success_count / true_rez, successes, jlocs]
    return [p, success_count / true_rez, successes]


class WorkspaceAnalyzer:
    """
    Used To Analyze A Variety of Pose Configurations for Defined Robots
    """

    def __init__(self, linkedRobot):
        self.bot = linkedRobot
        self.num_procs = 12
        if not self.bot.is_ready():
            print('Please Check Bindings')
        self.unique_decimals = UNIQUE_DECIMALS
        self.dof_offset = DOF_OFFSET
        self.alpha_value = ALPHA_VALUE
        self.max_dist = MAX_DIST

    def analyze_task_space(self, desired_poses):
        """
        Analyzes a cloud of potential poses, returns the percentage
            success of a robot to reach them.
        Args:
            desired_poses: Numpy NdArray of n 4x4 transformation matrices [n,4,4].
        Returns:
            Success: True if 100% of poses are reached successfully
            Percentage: Value 0-100 indicating percentage of poses reached
            Successful Poses: Numpy[i,4,4] NDArray indicating successful poses, None if none
            Failed Poses: Numpy [i,4,4] NdArray indicating failed poses, None if all are Successful
        """
        #TODO(Liam) add in swich to pull TM(matrix) from tm(object) if necessary
        #I don't think it should be necessary: can be otherwise specified in lambda funcs
        num_poses = len(desired_poses)
        failed_poses = []
        successful_poses = []
        for i in range(num_poses):
            desired_tm = desired_poses[i]
            success = self.bot.IK(desired_tm)
            if not success:
                failed_poses.append(desired_tm)
            else:
                successful_poses.append(desired_tm)
            progressBar(i, num_poses - 1, prefix='Analyzing Task Space')
        num_successful = len(successful_poses)
        num_failed = len(failed_poses)
        disp(num_successful, 'num_successful')
        disp(num_failed, 'num_failed')
        return num_successful, successful_poses

    def analyze_task_space_manipulability(self, desired_poses,
            manip_resolution=25, parallel=False, use_jacobian=False):
        """
        Given a cloud of points, determine manipulability at each point
        Args:
            desired_poses: cloud of desired points
            manip_resolution: approximate number of points on surface of unit sphere to try
            parallel: whether or not to use parallel computation
            use_jacobian: whether or not to use jacobian computation instead of unit sphere
        Returns:
            Results: [[Point, success_percentage, successful_orientations],[...],...]
        """

        num_poses = len(desired_poses)
        last_iter = num_poses - 1

        results = []

        sphr, _ = fsr.unitSphere(manip_resolution)
        sphr.append([0, 0, 0])
        sphere = np.array(sphr) * np.pi
        true_rez = len(sphere)

        start = time.time()
        if parallel and num_poses > 8 * self.num_procs:  # Is it *really* worth parallelism?
            async_results = []
            with mp.Pool(self.num_procs) as pool:
                for i in range(num_poses):
                    progressBar(i,
                            last_iter,
                            prefix='Spawning Async Tasks    ',
                            ETA=start)
                    async_results.append(pool.apply_async(process_point, (
                            desired_poses[i], sphere, true_rez, self.bot.robot,
                            False, use_jacobian)))
                start_2 = time.time()
                for i in range(num_poses):
                    progressBar(i,
                            last_iter,
                            prefix='Collecting Async Results',
                            ETA=start_2)
                    results.append(async_results[i].get(timeout=2*true_rez))
        else:

            for i in range(num_poses):
                progressBar(i,
                            last_iter,
                            prefix='Analyzing Cloud',
                            ETA=start)
                results.append(process_point(
                        desired_poses[i], sphere, true_rez, self.bot, False, use_jacobian))
        end = time.time()
        disp(end - start, 'Elapsed')
        return results


    def _total_workspace_recursive_helper(self, thetas_prior,
                                          iterations_per_dof, dof_iter):
        """
        Private Method Of Class To Help With Recursive Analysis of Workspace
        Args:
            thetas_prior: A partial list of joint thetas for use with the system
            iterations_per_dof: Number of iterations to match in the configuration space
            dof_iter: The current dof the system should be gneerating parameters for
        Returns:
            success_list: List of successful transformation matrices
        """
        success_list = []
        #disp(iterations_per_dof, "IterPerDof")
        # total_iter = iterations_per_dof * num_dof
        joint_configurations = np.linspace(self.bot.joint_mins[dof_iter],
                                           self.bot.joint_maxs[dof_iter],
                                           iterations_per_dof)
        for i in range(iterations_per_dof):
            theta_i = joint_configurations[i]
            theta_list = [theta_i] + thetas_prior

            #progressBar(total_iter-(iterations_per_dof*(dof_iter+1))+i, total_iter-1)
            #print(theta_list)
            if dof_iter == 0:
                #print(theta_list)
                ee_pos, success = self.bot.FK(np.array(theta_list))
                if success:
                    success_list.append(ee_pos)
            else:
                success_list.extend(
                    self._total_workspace_recursive_helper(
                        theta_list, iterations_per_dof, dof_iter - 1))
        return success_list

    def analyze_total_workspace_exhaustive_point_cloud(self,
                                                       num_iterations=12):
        """
        Analyzes total work_space of potential poses, returns numpy point cloud of reachable poses
        This is the basic method that returns
        Args:
            num_iterations: iterations at each dof
        Returns
            Reachability: Numpy point cloud
        """
        disp('Beginning Analysis')
        num_dof = len(self.bot.joint_mins)
        #iterations_per_dof = math.floor(num_iterations**(1/float(num_dof)))
        iterations_per_dof = num_iterations

        disp('Calibrating ETA')
        start = time.time()
        for _ in range(100):
            self.bot.FK(np.random.rand((num_dof)))
        end = time.time()
        end_estimate = end + (((end - start) / 100) *
                              ((num_dof - 1)**iterations_per_dof))
        end_str = datetime.fromtimestamp(end_estimate).strftime(
            '%m/%d/%Y, %H:%M:%S')
        disp('Analysis Estimated to Complete At: ' + end_str)
        disp('Starting')
        pose_cloud = self._total_workspace_recursive_helper([],
                                                            iterations_per_dof,
                                                            num_dof - 1)
        disp(
            'Analysis Completed At: ' +
            datetime.fromtimestamp(time.time()).strftime('%m/%d/%Y, %H:%M:%S'))
        return pose_cloud

    def _alpha_functional_workspace_recursive_helper(self, num_spread,
                                                     dof_iter, total_dof):
        """
        Helper function for alpha work_space analyzer tool.

        Should not be called individually.
        Args:
            num_spread - Number of samples to take at every DOF
            dof_iter - DOF to consider
            total_dof - Number of total degrees of freedom in the robot
        Returns:
            cloud of points reachable by robot / boundary region of points
        """
        success_list = []

        joint_configurations = np.linspace(self.bot.joint_mins[dof_iter],
                                           self.bot.joint_maxs[dof_iter],
                                           num_spread)
        theta_list = np.zeros((total_dof))
        if dof_iter == total_dof - 1:
            #If this is the final dof, we don't need to calculate the alpha shape
            disp('Calculating Iter: ' + str(dof_iter))
            for i in range(num_spread):
                theta_list[dof_iter] = joint_configurations[i]
                ee_pos, success = self.bot.FK(theta_list)
                if success:
                    success_list.append(ee_pos)
            return success_list
        else:
            current_cloud = self._alpha_functional_workspace_recursive_helper(
                num_spread, dof_iter + 1, total_dof)
            #disp(current_cloud[50:60])
            ee_pos, _ = self.bot.FK(theta_list)  #Reset to Neutral Pose

            local_cloud = [fsr.globalToLocal(ee_pos, x) for x in current_cloud]
            if dof_iter < (total_dof - self.dof_offset):
                point_list = [point.TAA.flatten()[0:3] for point in local_cloud]

                array_temp = np.array(point_list)
                array_filtered = np.unique(array_temp.round(decimals=self.unique_decimals), axis=0)
                #Could potentially implement the point exclusion section
                ashape = AlphaShape(array_filtered, self.alpha_value)
                verts = ashape.verts
                #local_cloud = []
                local_cloud = [tm([v[0], v[1], v[2], 0, 0, 0]) for v in verts]
                #for vert in verts:
                #    local_cloud.append(tm([vert[0], vert[1], vert[2], 0,0,0]))
            for i in range(num_spread):
                theta_list[dof_iter] = joint_configurations[i]
                ee_pos, success = self.bot.FK(theta_list)
                if success:
                    success_list.extend([fsr.localToGlobal(ee_pos, x) for x in local_cloud])

            disp('Calculating Iter: ' + str(dof_iter))
            return success_list

    def analyze_total_workspace_functional(self, num_spread=100):
        """
        Purpose: Analyze the total extent of a work_space of a serial manipulator without
            exhausting computational resources or exponential time for additional DOF.
            This method should be used to get a rough idea of the total manipulability of a robot
            in a comparatively short amount of time.
        Advantages:
            1) Overall iterations scale linearly, not exponentially, with added DOF to a robot
            2) Stores fewer points overall than an exhaustive search
        Limitations:
            1) This method does not, and can not account for force limitations, as it is predictive
                and not exhaustive.
            2) This method does not account for collisions, if they are possible on a robot.
            3) May only work for serial manipulators (Parallel untested/unknown)
        Idealized Proceedure:
            1) From the last joint to the first, each DOF is evaluated separately.
            2) Begin at the end effector and move the joint in an arc between its joint limits
            3) Record the position of the end effector relative to the transform of the joint
                in its neutral pose
            4) Return an alpha shape on the set of points, discounting those within a volume
            5) For each succesive dof, translate those points along with the known transforms
                and repeat stepsilon_vector 2-5 for each remaining joint
        Methodology Justification:
            The straightforward methodology (to iteratively manipulate each joint for each
            possible configuration) scales exponentially with each additional DOF, so that
            an overactuated robot may take an obscene amount of time to complete an anlysis.
            This method only scales linearly, so it may be of some benefit for 7DOF+ robots.
        Args:
            num_spread: the number of samples to draw from a single joint's motion range
        Returns:
            resultslist
        """
        num_dof = len(self.bot.joint_mins)
        return self._alpha_functional_workspace_recursive_helper(
            num_spread, 0, num_dof)

    def analyze_6dof_manipulability(self,
                                    shell_range=4,
                                    num_shells=30,
                                    points_per_shell=1000):
        """
        Analyze the extent of a work_space that can be reached in a full gamut of 6DOF orientations
        Idealized Proceedure:
            1. Generate a desired number of unit shells out from the origin of the robot.
                Each Unit Shell should have a desired surface density. These shells will
                represent test positions, devoid of rotation information.
                A) This can be augmented with the results of a prior total_work_space result
                    wherein points in the unit shell not within the total_worspace hull are
                    summarily discounted
            2. For each counted point (Regardless of implication of 1A) generate another unit
                sphere of a given density. This will serve as rotational information.
            3. For each combination of spheres A and B, the pose [A[0:3], B[0:3]] representing
                A 6DOF transformation will be generated, and the test arm's IK function called
            4. points will be color coded based on the percent reachability they possess.
        Methodology Justification:
            Despite IK being slower than FK for serial arms, this method does not scale
            exponentially with the addition of extra degrees of freedom and is constant for
            any type of serial manipulator, based on the desired number of iterations.

            Color coding manipulability regions, while not testing the overall work_space,
            can inform where regions are likely to have complete manipulability capabilities.
        Args:
            shell_range: maximum distance from origin of the farthest shell
            num_shells: number of shells
            points_per_shell: number of points in a shell
        Returns:
            transformation list of successful 6DOF poses.
        """
        shell_radii = np.linspace(0, shell_range, num_shells)
        shells = ([[[y * rad for y in x]
                    for x in fsr.unitSphere(points_per_shell)[0]]
                   for rad in shell_radii])
        scores = []
        i = 0
        for shell in shells:
            for point in shell:
                progressBar(i, (num_shells + 2) * points_per_shell,
                            prefix='Analyzing Reachability')
                i += 1

                def manipulability_at_point(p, r):
                    theta, suc = self.bot.IK(
                        tm([p[0], p[1], p[2], r[0], r[1], r[2]]))
                    if suc:
                        man_v, man_w = calculate_manipulability_score(
                            self.bot, theta)
                        if man_v == 0 or man_w == 0:
                            return -9999
                        else:
                            return 1 - (man_v + man_w) / 2
                    else:
                        return -999999

                initial_rot = np.zeros((3))
                _, succ = self.bot.IK(
                    tm([point[0], point[1], point[2], 0, 0, 0]))

                #botres = lambda x : manipulability_at_point(point, x)
                #sol = sci.optimize.fmin(botres, initial_rot, disp=False)
                sol = initial_rot
                score = manipulability_at_point(point, sol)
                if score < 0 or not succ:
                    continue
                scores.append([point, score])
        return scores
        #find a way to manipulate here for optimal reachability

    def analyze_manipulability_on_object_surface(self,
                                                 object_file_name,
                                                 object_scale=1,
                                                 object_pose=tm(),
                                                 manip_resolution=25,
                                                 collision_detect=True,
                                                 exempt_ee=True,
                                                 parallel=False,
                                                 use_jacobian=False,
                                                 minimum_dist=-1.0,
                                                 bound_shape=None,
                                                 approximate_bounding=False):
        """
        Analyze Manipulability On The Surface of An Object
        Given a file handle describing a STL mesh,
        this function examines each vertix for manipulability,
        and then prunes all configurations wherein some portion of the
        arm collides with the object.
        Args:
            object_file_name: name of the stl to be laoded
            object_scale: factor to resize stl by
            object_pose: global position to translate STL to
            manip_resolution: desired number of iterations per unit sphere
            collision_detect: whether or not to prune colliding configurations
            exempt_ee: exempt the end effector from collision detection (account for numerical acc)
            parallel: whether or not to use parallel processing to accelerate proceedure
            use_jacobian: Use Jacobian manipulability instead of unit sphere manipulability
            minimum_dist: Minimum distance between vertices for them to be considered
            bound_shape: bounding volume (optional) to further reduce points being considered
        Returns:
            results: list of visited points along with scores
            mesh: the loaded STL mesh
        """

        stl_mesh = mesh.Mesh.from_file(object_file_name)
        start = time.time()
        num_vertices_x = len(stl_mesh.x)

        for i in range(num_vertices_x):
            #TODO(LIAM) this can probably be optimized through numpy
            progressBar(i,
                        num_vertices_x - 1,
                        prefix='Preparing Mesh              ',
                        ETA=start)
            for j in range(3):
                mesh_pos_temp = tm([
                    stl_mesh.x[i, j] * object_scale,
                    stl_mesh.y[i, j] * object_scale,
                    stl_mesh.z[i, j] * object_scale, 0, 0, 0
                ])
                t_new = (object_pose @ mesh_pos_temp).TAA.flatten()
                stl_mesh.x[i, j] = t_new[0]
                stl_mesh.y[i, j] = t_new[1]
                stl_mesh.z[i, j] = t_new[2]

        points = np.unique(stl_mesh.vectors.reshape(
            [int(stl_mesh.vectors.size / 3), 3]),
                           axis=0)

        if bound_shape is not None:
            #Filter out everything we know the robot can't reach anyway
            filtered_points = inside_alpha_shape(bound_shape, points)
            if len(filtered_points) == 0:
                return [], stl_mesh

        sphr, _ = fsr.unitSphere(manip_resolution)
        num_points = len(points)
        sphr.append([0, 0, 0])
        sphere = np.array(sphr) * np.pi
        true_rez = len(sphere)
        results = []
        nd = len(self.bot.getJointTransforms())
        bot_base = self.bot.getJointTransforms()[0]

        joint_locs_arr_len = num_points * true_rez * nd
        verification_array = np.zeros((joint_locs_arr_len, 3))

        if num_points < 2 * self.num_procs:
            parallel = False  # Verify there's really a need for parallelism

        #num_points = len(vertices)
        #points = vertices
        if minimum_dist > 0:
            seen_points = []

        #Step One: Filter Out All points Beyond Maximum Reach of the Arm

        #Step Two: Calculate Manipulability on All points within Maximum reach of the Arm
        if parallel:
            with mp.Pool(self.num_procs) as pool:
                async_results = []
                start = time.time()
                for i in range(num_points):
                    progressBar(i,
                                num_points - 1,
                                prefix='Spawning Async Tasks        ',
                                ETA=start)
                    p = points[i, :]
                    #Ignore points we've already filtered out
                    if bound_shape is not None:
                        if p not in filtered_points:
                            async_results.append(
                                pool.apply_async(process_empty, (p, True)))
                            continue
                    #Ignore points which are too far away
                    if fsr.distance(p, bot_base) > self.max_dist:
                        #TODO(Liam) determine max reach from AShape
                        async_results.append(
                            pool.apply_async(process_empty, (p, True)))
                        continue
                    #Ignore points which are too close together
                    if minimum_dist > 0:
                        continuance = False
                        for point in seen_points:
                            if fsr.distance(point, p) < minimum_dist:
                                async_results.append(
                                    pool.apply_async(process_empty, (p, True)))
                                continuance = True
                                break
                        if continuance:
                            continue

                    async_results.append(
                        pool.apply_async(process_point,
                                         (p, sphere, true_rez, self.bot.robot,
                                          True, use_jacobian)))
                    if minimum_dist > 0:
                        seen_points.append(p)
                start = time.time()
                #print(num_points, len(async_results))
                for i in range(num_points):
                    progressBar(i,
                                num_points - 1,
                                prefix='Computing Reachability      ',
                                ETA=start)
                    results.append(async_results[i].get(timeout=2 * true_rez))
                start = time.time()
                for i in range(num_points):
                    progressBar(i,
                                num_points - 1,
                                prefix='Post Processing             ',
                                ETA=start)
                    if results[i][3] == []:
                        continue
                    for j in range(true_rez):
                        if results[i][3][j] == 0:
                            continue
                        for k in range(nd):

                            r = results[i]
                            ri = r[3]
                            rj = ri[j]
                            rk = rj[k]
                            sample_ind = (i * true_rez * nd) + (j * nd) + k
                            verification_array[sample_ind, :] = rk[0:3].flatten()
        else:
            for i in range(num_points):
                progressBar(i,
                            num_points - 1,
                            prefix='Computing Reachability      ',
                            ETA=start)
                p = points[i, :]
                #Ignore points we've already filtered out
                if bound_shape is not None:
                    if p not in filtered_points:
                        results.append(process_empty(p, True))
                        continue
                #Ignore points which are too far away
                if fsr.distance(p, bot_base) > self.max_dist:
                    results.append(process_empty(p, True))
                    continue
                #Ignore ponts which are too close together
                if minimum_dist > 0:
                    continuance = False
                    for point in seen_points:
                        if fsr.distance(point, p) < minimum_dist:
                            results.append(process_empty(p, True))
                            continuance = True
                            break
                    if continuance:
                        continue
                pdone = process_point(p, sphere, true_rez, self.bot, True,
                                      use_jacobian)
                if minimum_dist > 0:
                    seen_points.append(p)
                if pdone[3] == []:
                    continue
                for j in range(true_rez):
                    if pdone[3][j] == 0:
                        continue
                    for k in range(nd):
                        sample_index = (i * true_rez * nd) + (j * nd) + k
                        temp_val = pdone[3][j][k][0:3].flatten()
                        verification_array[sample_index, :] = temp_val
                results.append(pdone)

        if not collision_detect:
            return results, stl_mesh

        #Step Three: Locate Instances Where the Arm Collides WIth The Object In Question
        tri_vecs = None

        if approximate_bounding:
            if bound_shape is not None:
                tri_vecs = np.array(AlphaShape(filtered_points, self.alpha_value).triangles)
            else:
                tri_vecs = np.array(AlphaShape(points, self.alpha_value).triangles)
        else:
            tri_vecs = stl_mesh.vectors
        vec_len = len(tri_vecs)

        def isodd(x):
            return x % 2 != 0

        num_intersections = np.zeros(joint_locs_arr_len)
        if parallel:
            with mp.Pool(self.num_procs) as pool:
                # Integer division to determine chunk size to feed to parallel computation
                # Excess is absorbed by the last chunk
                chunk_sz = vec_len // self.num_procs
                async_startup = []
                for i in range(self.num_procs):
                    progressBar(i, self.num_procs - 1,
                                'Initializing Moller Trumbore')
                    if i == self.num_procs - 1:
                        async_startup.append(
                            pool.apply_async(chunk_moller_trumbore,
                                             (verification_array,
                                              tri_vecs[i * chunk_sz:, :],
                                              np.array([EPSILON, 0, RAY_UNIT - EPSILON]),
                                              np.zeros(joint_locs_arr_len))))
                    else:

                        async_startup.append(
                            pool.apply_async(
                                chunk_moller_trumbore,
                                (verification_array,
                                 tri_vecs[i * chunk_sz:(i + 1) *
                                                  chunk_sz, :],
                                 np.array([EPSILON, 0, RAY_UNIT - EPSILON
                                           ]), np.zeros(joint_locs_arr_len))))

                waiter = WaitText('Running Moller Trumbore')
                while not async_startup[0].ready():
                    waiter.print()
                    time.sleep(.25)
                waiter.done()
                moller_results = []
                start = time.time()
                for i in range(self.num_procs):
                    progressBar(i,
                                self.num_procs - 1,
                                'Retrieving Results          ',
                                ETA=start)
                    moller_results.append(async_startup[i].get(timeout=100))
                print('Combining Results...')
                for i in range(self.num_procs):
                    #print(np.shape(moller_results[i]))
                    num_intersections += moller_results[i]
        else:
            start = time.time()
            for i in range(vec_len):
                progressBar(i,
                            vec_len - 1,
                            prefix='Executing Moller Trumbore   ',
                            ETA=start)
                res = moller_trumbore_ray_intersection_array(
                    verification_array, tri_vecs[i, :],
                    np.array([EPSILON, 0, RAY_UNIT - EPSILON]))

                # where(res == True)
                num_intersections[np.where(res)] += 1

        # Step Four: Discard Instances Where The Arm Collides With The Object In Question
        lva = len(verification_array)
        start = time.time()
        penalized_configs = []
        for i in range(lva):
            progressBar(i,
                        lva - 1,
                        prefix='Filtering out Collisions    ',
                        ETA=start)
            if isodd(num_intersections[i]):
                #sample_index = (i*true_rez*nd)+(j*nd)+k For reference
                top_ind = i // (true_rez * nd)  # Top Level Bucket
                snd_ind = i // (nd)  # Second Level Bucket
                if exempt_ee and snd_ind == nd - 1:
                    #TODO(Liam) is this correct?
                    continue

                config_duo = (top_ind, snd_ind)
                if config_duo in penalized_configs:
                    continue
                penalized_configs.append(config_duo)
                if sum(abs(results[top_ind][0])) < .001:
                    #Is near zero
                    continue

                results[top_ind][1] -= (1 / true_rez)
        return results, stl_mesh

    def analyze_manipulability_within_volume(self,
                                             shape,
                                             grid_resolution=.25,
                                             manip_resolution=25,
                                             parallel=False,
                                             use_jacobian=False):
        """
        Given a volume encapsulating a work_space, analyze the 6DOF positioning capability within
        That particular work_space.
        Args:
            shape: instance of AlphaShape class, representing the shape to be analyzed
            grid_resolution: the resolution of the grid to be
                created (distance between points in meters)
            manip_resolution: the target number of points to create on a unit sphere
                to measure rotational capability at a given point
            parallel: analyze in parallel
            use_jacobian: use jacobian manipulability optimization instead
        Returns:
            scorelist: a list containing points, scores, and successful positions:
                each element takes the form: [p, s, posl]
                p: the target point
                s: the score 0-1 of manipulability at the point. 1 is best
                posl: the list of successful pose configurations at that point
        """

        point_cloud = grid_cloud_within_volume(shape, grid_resolution)

        return self.analyze_task_space_manipulability(point_cloud,
                manip_resolution, parallel, use_jacobian)

    def analyze_manipulability_over_trajectory(self,
                                               point_list,
                                               manipulability_mode=1,
                                               manip_resolution=25,
                                               point_interpolation_dist=-1,
                                               point_interpolation_mode=1):
        """
        Given a list of points to test, analyze the manipulability over the trajectory
        Args:
            point_list: ([tm]) waypoint list comprising the trajectory. Minimum of 2 points
            manipulability_mode: [Optional Int]
                1- use Jacobian Manipulability (eigenvalues)
                2- use Unit Sphere Manipulability (calculate various orientations at each point)
            manip_resolution: [Optional Int] number of points to set up unit sphere.
                Only used if manipulability_mode is not 1
            point_interpolation_dist: [Optional Int] if > 0 will autopopulate trajectory
                with filler points between waypoints using mode specified
                in point_interpolation_mode
            point_interpolation_dist: [Optional Int] sets the method of interpolation
                1- Use Linear (Gap Closure) interpolation
                2- Use Arc (Arc Gap) interpolation
        Returns:
            results_list: [(point, score)] results list, one entry per point in trajectory
        """

        #Step One: Flesh out the point list if necessary
        if point_interpolation_dist > 0:
            new_points = []
            point_list_len = len(point_list)
            start = time.time()
            for i in range(point_list_len):
                progressBar(i,
                            point_list_len - 1,
                            'Filling Trajectory      ',
                            ETA=start)
                if i == point_list_len - 1:
                    new_points.append(point_list[i])
                    continue
                start_point = point_list[i]
                end_point = point_list[i + 1]
                distance = fsr.arcDistance(start_point, end_point)
                if distance < point_interpolation_dist:
                    continue
                while distance > point_interpolation_dist:
                    if point_interpolation_mode == 1:
                        start_point = fsr.closeLinearGap(start_point, end_point,
                                                   point_interpolation_dist)
                    else:
                        start_point = fsr.closeArcGap(start_point, end_point,
                                                 point_interpolation_dist)
                    distance = fsr.arcDistance(start_point, end_point)
                    new_points.append(start_point)
            point_list = new_points

        #Step Two: Determine Manipulability Function
        results = self.analyze_task_space_manipulability(point_list,
                manip_resolution, False, manipulability_mode == 1)

        return results
