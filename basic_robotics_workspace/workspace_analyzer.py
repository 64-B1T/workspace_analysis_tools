# Utility Imports
from datetime import datetime
import multiprocessing as mp
import numpy as np
import os
import scipy as sci
import time

#This module imports
import workspace_constants
from alpha_shape import AlphaShape
from workspace_helper_functions import wait_for, complete_trajectory
from workspace_helper_functions import ignore_close_points, gen_manip_sphere
from workspace_helper_functions import grid_cloud_within_volume
from workspace_helper_functions import process_empty

from moller_trumbore import inside_alpha_shape

# Other module imports
from basic_robotics.robot_collisions import createMesh, ColliderManager
from basic_robotics.robot_collisions import ColliderArm, ColliderObstacles
from basic_robotics.general import tm, fsr
from basic_robotics.utilities.disp import disp, progressBar

# Helper Functions That You Probably Shouldn't Need To Call Directly
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
        # Calculate manipulability with given orientation of end effector
        theta, suc = bot.IK(pos)
        if not suc:
            return 0
        mrv, mrw = calculate_manipulability_score(bot, theta)
        return (mrv + mrw) / 2

    def lambda_handle(x, pos=pos):
        # Handle for the maximize helper to be plugged into fmin
        return 1 - maximize_helper(tm([pos[0], pos[1], pos[2], x[0], x[1], x[2]]))

    rot_init = np.zeros(3)  # Initialize to no orientation
    sol = sci.optimize.fmin(lambda_handle, rot_init, disp=False)
    max_manip_pos = tm([pos[0], pos[1], pos[2], sol[0], sol[1], sol[2]])
    theta, suc = bot.IK(max_manip_pos)
    if not suc:
        return 0, tm(), theta  # If the chosen position wasn't successful, there probably isn't
        #   a potential solution at all, so just return an empty set
    score = sum(calculate_manipulability_score(bot, theta)) / 2
    return score, max_manip_pos, theta


def calculate_manipulability_score(bot, theta):
    """
    Calculate the manipulability ellipsoid at a given pose.

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

    #Extract the eigenvectors
    aw_eig, _ = np.linalg.eig(a_w)
    av_eig, _ = np.linalg.eig(a_v)

    manipulability_ratio_w = 1 / (np.sqrt(max(aw_eig)) / np.sqrt(min(aw_eig)))
    #disp(manipulability_ratio_w)
    manipulability_ratio_v = 1 / (np.sqrt(max(av_eig)) / np.sqrt(min(av_eig)))
    #disp(manipulability_ratio_v)
    return manipulability_ratio_v, manipulability_ratio_w

def get_collision_data(collision_manager):
    """
    Helper function for determining if a collision is ocurring
    Args:
        collision_manager: ColliderManager object populated with an arm and optional obstacle
    Returns:
        boolean for collision
    """
    collision_manager.update()
    inter_collisions = False
    if len(collision_manager.collision_objects) > 1:
        # if there's another object in the collision manager, it's likely an obstacle
        #   that we need to consider.
        inter_collisions = collision_manager.checkCollisions()[0]
    solo_collisions = collision_manager.collision_objects[0].checkInternalCollisions()
    if inter_collisions or solo_collisions:
        return True
    return False

def process_point(p,
                  sphere,
                  true_rez,
                  bot,
                  use_jacobian=False,
                  collision_detect=False,
                  collision_manager=None):
    """
    Process a point against a unit sphere for success metric.

    Args:
        p: Point to process
        sphere: unit sphere
        true_rez: the true amount of points inside the unit sphere
        bot: reference to the robot object
        use_jacobian: [Optional Bool] Use Jacobian instead of unit sphere
        collision_detect: [Optional Bool] detect collisions (Default False)
        collision_manager: [Optional ColliderManager] provide if setting collision Detect on
    Returns:
        Results: [point, success_percentage, successful_orientations, successful_thetas]
    """
    successes = []
    thetas = []
    success_count = 0
    if use_jacobian:
        score, max_pos, theta = maximize_manipulability_at_point(bot, p)
        if collision_detect and get_collision_data(collision_manager):
            score = 0
        if score == 0:
            return [p, 0, [], []]
        return [p, score, [max_pos], [theta]]
    for j in range(true_rez):
        rot = sphere[j, :]
        goal = tm([p[0], p[1], p[2], rot[0], rot[1], rot[2]])
        theta, suc = bot.IK(goal)
        if collision_detect and get_collision_data(collision_manager):
            suc = False
        if suc:
            thetas.append(theta)
            successes.append(rot)
            success_count += 1
    return [p, success_count / true_rez, successes, thetas]

def setup_collision_manager(bot, stl_mesh = None, exempt_ee = False):
    """
    Set up a collision manager for a robot arm and optional object

    Because the collision manager, being C based, is not pickleable,
    it must be set up fresh in each subprocess which wishes to use it.

    Args:
        bot: robot to use inside of the collision manager
        stl_mesh: optional mesh file detailing an obstacle that the robot must avoid
        exempt_ee: whether or not to consider the EE of the robot in collisions

    Returns:
        ColliderManager: Collision manager, with arm in index 0
    """
    collision_manager = ColliderManager()
    collision_manager.bind(ColliderArm(bot, 'model'))
    if stl_mesh is not None:
        obstacle = ColliderObstacles('object_surface')
        obstacle.addMesh('object', stl_mesh)
        collision_manager.bind(obstacle)
    if exempt_ee:
        collision_manager.collision_objects[0].deleteEE()
        collision_manager.collision_objects[0].deleteEE()
    return collision_manager

def chunk_point_processing(points, sphere, true_rez, bot,
        use_jacobian, collision_detect, stl_mesh=None,
        exempt_ee=False, display_eta=False):
    """
    Process points in a chunk (for parallel computation)

    Args:
        points: list of points to process
        sphere: unit sphere to use
        true_rez: true resolution of the unit sphere
        bot: reference to robot object
        use_jacobian: bool to use a jacobian
        collision_detect: bool to detect collisions
        stl_mesh: optional stl mesh object to use for collision checking
        exempt_ee: optional exemption of end effector from collision checking
        display_eta: optional boolean to display a progress bar

    Returns:
        list: results (as of process point)

    """
    collision_manager = None
    if collision_detect:
        collision_manager = setup_collision_manager(bot, stl_mesh, exempt_ee)
    results = []
    last_iter = len(points)
    start = time.time()
    for i in range(last_iter):
        results.append(process_point(points[i], sphere,
                true_rez, bot, use_jacobian,
                collision_detect, collision_manager))
        if display_eta:
            progressBar(i,
                    last_iter - 1,
                    prefix='Chunked Point Processing',
                    ETA=start)
    return results


def parallel_brute_manipulability_collision_head(
        bot, thetas_prior, resolution, excluded,
        dof_iter, collision_detect):
    """
    Head function to set up brute manipulability/reachability analysis in parallel
    Args:
        bot: Robot object
        thetas_prior: prior thetas (usually going to be an empty list)
        resolution: number of positions to interpolate per point
        excluded: list of joints to exclude from conisderation (can be empty)
        dof_iter: current iteration of the joints of the robot to consider
        collision_detect: Boolean to enable collision detection
    Returns:
        list: results list
    """

    collision_manager = None
    if collision_detect:
        collision_manager = setup_collision_manager(bot)
    results = brute_fk_manipulability_recursive_process(bot,
        thetas_prior, resolution, excluded, dof_iter,
        time.time(), collision_detect, collision_manager)
    return results


def brute_fk_manipulability_recursive_process(bot, thetas_prior,
        resolution, excluded, dof_iter, start, collision_detect, collision_manager):
    """
    Brute force Forward Kinematics based manipulability analysis

    Args:
        bot: Robot object
        thetas_prior: prior thetas (usually going to be an empty list)
        resolution: number of positions to interpolate per point
        excluded: list of joints to exclude from conisderation (can be empty)
        dof_iter: current iteration of the joints of the robot to consider
        collision_detect: Boolean to enable collision detection
        start: start time
        collision_detect: Boolean to enable collision detection
        collision_manager: Collision Manager object
    Returns:
        list: results list

    """
    success_list = []
    if dof_iter in excluded:  #To run full brute manipulability, simply pass in an empty list
        if dof_iter > 0:
            theta_list = [0.0] + thetas_prior
            return brute_fk_manipulability_recursive_process(bot,
                    theta_list, resolution, excluded, dof_iter - 1, start,
                    collision_detect, collision_manager)
        #If this is the base link, we have no choice but to process with zeros
        joint_configurations = np.zeros(resolution)
    else:
        joint_configurations = np.linspace(
            bot.joint_mins[dof_iter],
            bot.joint_maxs[dof_iter],
            resolution)
    for i in range(resolution):
        if dof_iter == len(bot.joint_mins) - 1:
            # If this is the top layer of recursion, we can run a progress bar for the user
            progressBar(i, resolution-1, 'Running Brute Manipulability', ETA=start)
        theta_i = joint_configurations[i]
        theta_list = [theta_i] + thetas_prior
        if dof_iter == 0:  # Calculations only occur at the bottom recursion layer
            theta_array = np.array(theta_list)
            ee_pos = bot.FK(theta_array)
            if isinstance(ee_pos, tuple):
                #This is here to handle the case in which parallelism is applied and the
                #   Robot link is not used
                ee_pos = ee_pos[0]
            if collision_detect:
                collision_manager.update()
                if collision_manager.checkCollisions()[0]:
                    # If in collision, don't bother to check manipulability
                    continue
            mrv, mrw = calculate_manipulability_score(bot, theta_array)
            score = (mrv + mrw) / 2
            result = [ee_pos, score, theta_array, [mrv, mrw]]
            success_list.append(result)
        else:
            #If not the base link, simply recurse
            success_list.extend(
                brute_fk_manipulability_recursive_process(bot,
                        theta_list, resolution, excluded,
                        dof_iter - 1, start,
                        collision_detect, collision_manager))
    return success_list

class WorkspaceAnalyzer:
    """
    Used To Analyze A Variety of Pose Configurations for Defined Robots
    """

    def __init__(self, linkedRobot):
        """
        Create a workspace analyzer object which can be implemented by other software
        Args:
            linkedRobot: Linked robot to analyze
        Returns:
            WorkspaceAnalyzer: Resulting analysis handle
        """
        self.bot = linkedRobot
        cpu_count = os.cpu_count()
        self.num_procs = 4
        if cpu_count is not None:
            self.num_procs = cpu_count - 1  # Leave a CPU for doing other things
        if not self.bot.is_ready():
            print('Please Check Bindings')
        self.unique_decimals = workspace_constants.UNIQUE_DECIMALS
        self.dof_offset = workspace_constants.DOF_OFFSET
        self.alpha_value = workspace_constants.ALPHA_VALUE
        self.max_dist = workspace_constants.MAX_DIST

    def parallel_process_point_cloud(self, num_poses, desired_poses, sphere, true_rez,
            use_jacobian=False, collision_detect=False, stl_mesh=None, exempt_ee=False):
        """
        Run a point cloud through parallel processing, potentially with collision checking

        Args:
            num_poses: number of points to calculate
            desired_poses: points to calculate
            sphere: unit sphere to use
            true_rez: true resolution of the unit sphere
            use_jacobian: bool to use a jacobian
            collision_detect: bool to detect collisions
            stl_mesh: optional stl mesh object to use for collision checking
            exempt_ee: optional exemption of end effector from collision checking

        Returns:
            Results: [[point, success_percentage, successful_orientations, successful_thetas],[]]
        """
        async_results = []
        results = []
        start = time.time()
        chunk_sz = num_poses // self.num_procs
        with mp.Pool(self.num_procs) as pool:
            for i in range(self.num_procs):
                progressBar(i,
                        self.num_procs - 1,
                        prefix='Spawning Async Tasks        ',
                        ETA=start)
                display_eta = i == (self.num_procs - 1)
                small_ind = i * chunk_sz
                large_ind = (i + 1) * chunk_sz
                if display_eta:
                    large_ind = num_poses
                async_results.append(pool.apply_async(chunk_point_processing, (
                        desired_poses[small_ind:large_ind], sphere, true_rez, self.bot.robot,
                        use_jacobian, collision_detect, stl_mesh, exempt_ee, display_eta)))
            wait_for(async_results[-1], 'Running Point Detection    ')
            start_2 = time.time()
            for i in range(self.num_procs):
                progressBar(i,
                        self.num_procs - 1,
                        prefix='Collecting Async Results',
                        ETA=start_2)
                results.extend(async_results[i].get(timeout=5*true_rez))
        return results

    def analyze_joint_related_to_end_effector_vals(self, manipulability_space, joint_goal_norm, angular = False):
        """
        Calculates the minimum joint values to relate to desired end effector norm
        Works for velocities and torques

        Args:
            manipulability_space: results from a manipulability analysis
            joint_goal_norm: goal joint norm

        Results:
            augmented results with joint vals added at each point
        """
        inds = [3, 6] # Extracts the linear component of a wrench
        if angular:
            inds = [0, 3] # Extracts the angular component of a wrench
        def minimize_ee_norm(joint_vals):
            ee_vals = self.bot.robot._jointsToEndEffectorJacobian(joint_vals).flatten()
            new_norm = np.linalg.norm(ee_vals[inds[0]:inds[1]])
            if new_norm > joint_goal_norm:
                return new_norm - joint_goal_norm
            else:
                return joint_goal_norm + new_norm
        results = []
        xs = sci.optimize.fmin(minimize_ee_norm, manipulability_space[0][3][0], disp = False)
        for c in manipulability_space:
            res_sub = []
            for joint_config in c[3]:
                self.bot.FK(joint_config)
                xs = sci.optimize.fmin(minimize_ee_norm, xs, disp = False)
                res_sub.append([joint_config, xs])
            results.append([c[0], res_sub])
        return results

    def analyze_matching_joint_torques(self, manipulability_space,
            mass_cg, mass, grav_vector = np.array([0, 0, -9.8])):
        """
        Calculate joint torques for a given end effector wrench

        Args:
            manipulability_space: results list from earlier manipulability analysis
            mass_cg: center of gravity tm for the applied mass relative to EE (global)
            mass: mass in kilograms of payload
            grav_vector: applied gravity. Normally this would be [0, 0, -9.8] default

        Returns:
            list: list corresponding joint torques and thetas to points in manipulability space

        """
        results = []
        for c in manipulability_space:
            res_sub = []
            for joint_config in c[3]:
                self.bot.FK(joint_config)
                ee_wrench = fsr.makeWrench(self.bot.getEE() @ mass_cg, mass, grav_vector)
                torques = self.bot.robot.staticForceWithLinkMasses(joint_config, ee_wrench)
                res_sub.append([joint_config, torques])
            results.append([c[0], res_sub])
        return results

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
            manip_resolution=25, parallel=False, use_jacobian=False, collision_detect=False):
        """
        Given a cloud of points, determine manipulability at each point
        Args:
            desired_poses: cloud of desired points
            manip_resolution: approximate number of points on surface of unit sphere to try
            parallel: whether or not to use parallel computation
            use_jacobian: whether or not to use jacobian computation instead of unit sphere
            collision_detect: [Optional Bool] detect collisions (Default False)
        Returns:
            Results: [[Point, success_percentage, successful_orientations],[...],...]
        """

        num_poses = len(desired_poses)
        last_iter = num_poses - 1
        results = []

        sphere, true_rez = gen_manip_sphere(manip_resolution)
        collision_manager = None

        start = time.time()
        if parallel and num_poses > 8 * self.num_procs:  # Is it *really* worth parallelism?
            results = self.parallel_process_point_cloud(num_poses, desired_poses, sphere, true_rez,
                    use_jacobian=use_jacobian, collision_detect=False)
        else:
            if collision_detect:
                collision_manager = setup_collision_manager(self.bot)
            for i in range(num_poses):
                progressBar(i,
                            last_iter,
                            prefix='Analyzing Cloud',
                            ETA=start)
                results.append(process_point(
                        desired_poses[i], sphere, true_rez, self.bot,
                        use_jacobian, collision_detect, collision_manager))
        end = time.time()
        disp(end - start, 'Elapsed')
        return results

    def _parallel_brute_manipulability_recursive_head(
            self, thetas_prior, resolution, excluded, dof_iter, collision_detect):
        """
        Run brute recursive manipulability analysis in parallel.

        Args:
            thetas_prior: prior thetas (usually going to be an empty list)
            resolution: number of positions to interpolate per point
            excluded: list of joints to exclude from conisderation (can be empty)
            dof_iter: current iteration of the joints of the robot to consider
            collision_detect: Boolean to enable collision detection

        Returns:
            List: results list

        """

        #If the resolution is lower than number of processors, probably not worth it.
        if resolution < self.num_procs:
            return self.analyze_brute_manipulability_on_joints(
                    resolution, excluded, parallel=False,
                    collision_detect=collision_detect)

        #If the end effector is excluded, we don't have to segment on it.
        if dof_iter in excluded:
            theta_list = [0.0] + thetas_prior
            return self._parallel_brute_manipulability_recursive_head(
                    theta_list, resolution, excluded, dof_iter - 1,
                    collision_detect)

        #If not, now we partition
        joint_configurations = np.linspace(
            self.bot.joint_mins[dof_iter],
            self.bot.joint_maxs[dof_iter],
            resolution)

        #Determine how to evenly chunk tasks
        start = time.time()
        async_results = []
        results = []
        with mp.Pool(self.num_procs) as pool:
            for i in range(resolution):
                theta_i = joint_configurations[i]
                theta_list = [theta_i] + thetas_prior
                progressBar(i,
                        resolution - 1,
                        prefix='Spawning Async Tasks        ',
                        ETA=start)
                async_results.append(pool.apply_async(
                    parallel_brute_manipulability_collision_head,
                    (self.bot.robot, theta_list, resolution,
                    excluded, dof_iter - 1, collision_detect)
                ))
            wait_for(async_results[-1], 'Running Brute Collection in Parallel')
            start_2 = time.time()
            for i in range(resolution):
                progressBar(i,
                        resolution - 1,
                        prefix='Collecting Async Results',
                        ETA=start_2)
                results.extend(async_results[i].get(timeout=10000))
            return results

    def analyze_brute_manipulability_on_joints(self, resolution, joint_indexes,
            parallel=False, collision_detect=False):
        """
        Run manipulability analysis on joints using a brute forced, FK approach

        Args:
            resolution: number of positions to interpolate per point
            excluded: list of joints to exclude from conisderation (can be empty)
            parallel: process in parallel as much as possible
            collision_detect: Boolean to enable collision detection

        Returns:
            List: results list

        """
        num_dof = len(self.bot.joint_mins)
        disp('Starting')
        total_iters = resolution ** (num_dof - len(joint_indexes))
        disp('Anticipated Iterations: ' + str(total_iters))
        if parallel:
            results = self._parallel_brute_manipulability_recursive_head(
                [], resolution, joint_indexes, num_dof - 1, collision_detect)
        else:
            collision_manager = None
            if collision_detect:
                collision_manager = setup_collision_manager(self.bot)
            results = brute_fk_manipulability_recursive_process(self.bot,
                [], resolution, joint_indexes, num_dof - 1,
                time.time(), collision_detect, collision_manager)
        disp('complete')
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
        joint_configurations = np.linspace(self.bot.joint_mins[dof_iter],
                                           self.bot.joint_maxs[dof_iter],
                                           iterations_per_dof)
        for i in range(iterations_per_dof):
            theta_i = joint_configurations[i]
            theta_list = [theta_i] + thetas_prior
            if dof_iter == 0:
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
                try:
                    point_list = [point.TAA.flatten()[0:3] for point in local_cloud]

                    array_temp = np.array(point_list)
                    array_filtered = np.unique(array_temp.round(decimals=self.unique_decimals), axis=0)
                    # Could potentially implement the point exclusion section
                    ashape = AlphaShape(array_filtered, self.alpha_value)
                    verts = ashape.verts
                    local_cloud = [tm([v[0], v[1], v[2], 0, 0, 0]) for v in verts]
                except:
                    # Failure in case of insufficient dimensionality of arm will result in the alpha shape
                    # Generator to fail, 3
                    print("Insufficient Dimensions in Arm, Trying Next DOF")
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
                                    shell_range=4, # Distance from origin in meters to extend shell
                                    num_shells=30, # Number of shells to interpolate in shell range
                                    points_per_shell=1000, # Approximate number of points per shell
                                    collision_detect=False): # Utilize collision detection
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
            collision_detect: [Optional Bool] detect collisions (Default False)
        Returns:
            transformation list of successful 6DOF poses.
        """
        shell_radii = np.linspace(0, shell_range, num_shells)
        shells = ([[[y * rad for y in x]
                    for x in fsr.unitSphere(points_per_shell)[0]]
                   for rad in shell_radii])
        results = []
        i = 0
        collision_manager = None
        if collision_detect:
            collision_manager = setup_collision_manager(self.bot)
        for shell in shells:
            for point in shell:
                progressBar(i, (num_shells + 2) * points_per_shell,
                            prefix='Analyzing Reachability')
                i += 1
                results.append(process_point(point, None, None, self.bot,
                        True, collision_detect, collision_manager))
        return results
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
                                                 bound_shape=None):
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

        stl_mesh = createMesh(tm(), object_file_name)
        stl_mesh.vertices = stl_mesh.vertices * object_scale
        stl_mesh.apply_transform(object_pose.gTM())

        raw_points = []
        for i in range(len(stl_mesh.vertices)):
            raw_points.append(stl_mesh.vertices[i, :])
        points = np.unique(np.array(raw_points), axis=0)

        if bound_shape is not None:
            #Filter out everything we know the robot can't reach anyway
            filtered_points = inside_alpha_shape(bound_shape, points)
            if len(filtered_points) == 0:
                return [], stl_mesh

        num_points = len(points)
        sphere, true_rez = gen_manip_sphere(manip_resolution)
        results = []
        bot_base = self.bot.getJointTransforms()[0]

        if num_points < 2 * self.num_procs:
            parallel = False  # Verify there's really a need for parallelism

        #num_points = len(vertices)
        #points = vertices
        if minimum_dist > 0:
            seen_points = []

        if collision_detect:
            collision_manager = setup_collision_manager(self.bot, stl_mesh, exempt_ee)

        #Step One: Filter Out All points Beyond Maximum Reach of the Arm
        empty_results = []
        true_points = []
        start = time.time()
        for i in range(num_points):
            progressBar(i,
                        num_points - 1,
                        prefix='Preprocessing Data          ',
                        ETA=start)
            p = points[i, :]
            #Ignore points we've already filtered out
            if bound_shape is not None:
                if p not in filtered_points:
                    empty_results.append(process_empty(p))
                    continue
            #Ignore points which are too far away
            if fsr.distance(p, bot_base) > self.max_dist:
                empty_results.append(process_empty(p))
                continue
            #Ignore points which are too close together
            if minimum_dist > 0:
                continuance, empty_results = ignore_close_points(
                        seen_points, empty_results, p, minimum_dist)
                if continuance:
                    continue
            true_points.append(p)
            if minimum_dist > 0:
                seen_points.append(p)

        #Step Two: Calculate Manipulability on All points within Maximum reach of the Arm
        if parallel:
            results = self.parallel_process_point_cloud(len(true_points), true_points,
                    sphere, true_rez, use_jacobian,
                    collision_detect, stl_mesh, exempt_ee)
        else:
            for i in range(len(true_points)):
                p = true_points[i]
                progressBar(i,
                            num_points - 1,
                            prefix='Computing Reachability      ',
                            ETA=start)
                pdone = process_point(p, sphere, true_rez, self.bot, use_jacobian,
                        collision_detect, collision_manager)
                results.append(pdone)
        results.extend(empty_results)
        return results, stl_mesh

    def analyze_manipulability_within_volume(self,
                                             shape,
                                             grid_resolution=.25,
                                             manip_resolution=25,
                                             parallel=False,
                                             use_jacobian=False,
                                             collision_detect=False):
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
            use_jacobian: use jacobian manipulability optimization instead'
            collision_detect: [Optional Bool] detect collisions (Default False)
        Returns:
            scorelist: a list containing points, scores, and successful positions:
                each element takes the form: [p, s, posl]
                p: the target point
                s: the score 0-1 of manipulability at the point. 1 is best
                posl: the list of successful pose configurations at that point
        """

        point_cloud = grid_cloud_within_volume(shape, grid_resolution)

        return self.analyze_task_space_manipulability(point_cloud,
                manip_resolution, parallel, use_jacobian, collision_detect)

    def analyze_manipulability_point_cloud_with_trajectory(self, point_cloud, trajectory_list,
        manip_resolution, point_interpolation_dist=-1, manipulability_mode=True,
        point_interpolation_mode=1, collision_detect=False, parallel=False):
        """
        Analyze manipulability on a point cloud with trajectory applied to each point.
        Make sure that the initial points are oriented correctly for the analysis to work.

        Args:
            point_cloud: Point cloud (perhaps representing the surface of an object)
            trajectory_list: the local waypoints that make up a trajectory (such as withdrawing)
            manip_resolution: manipulation resultion of a unit sphere
            point_interpolation_dist: maximum distance between interpolated points in a trajectory
            manipulability_mode: whether or not to use jacobian or unit sphere manipulabilitity
            point_interpolation_mode: Whether or not to use arc interpolation
            collision_detect: Whether or not to use collision detection
            parallel: Whether or not to evaluate in parallel

        Returns:
            List: Results List for trajectory analysis
        """

        print('Interpolating Trajectory As Required')
        trajectory_list = complete_trajectory(
            trajectory_list, point_interpolation_dist, point_interpolation_mode)

        print('Preprocessing requested local trajectory')
        trajectory_cloud = []

        for point in point_cloud:
            local_trajectory = []
            for traj_point in trajectory_list:
                local_trajectory.append(fsr.globalToLocal(point, traj_point))
            trajectory_cloud.append(local_trajectory)

        complete_results = []
        results = []
        #It's probably better to handle parallelism on a surface level here
        if parallel:
            async_results = []
            chunk_sz = len(trajectory_cloud) // self.num_procs
            with mp.Pool(self.num_procs) as pool:
                start = time.time()
                results = []

                sphere, true_rez = gen_manip_sphere(manip_resolution)
                for i in range(self.num_procs):
                    progressBar(i,
                            self.num_procs - 1,
                            prefix='Spawning Async Tasks        ',
                            ETA=start)
                    display_eta = i == (self.num_procs - 1)
                    small_ind = i * chunk_sz
                    large_ind = (i + 1) * chunk_sz
                    if display_eta:
                        large_ind = len(trajectory_cloud)
                    for t in trajectory_cloud[small_ind:large_ind]:
                        async_results.append(pool.apply_async(chunk_point_processing, (
                                t, sphere, true_rez, self.bot.robot,
                                manipulability_mode, collision_detect
                                )))
                wait_for(async_results[-1], 'Running Task Manipulability')
                start_2 = time.time()
                for i in range(len(async_results)):
                    progressBar(i,
                            len(async_results)-1,
                            prefix='Collecting Async Results',
                            ETA=start_2)
                    results.append(async_results[i].get(timeout=5*len(trajectory_cloud)))

        else:
            for trajectory in trajectory_cloud:
                #Evaluate the trajectory
                trajectory_result = self.analyze_task_space_manipulability(trajectory,
                        manip_resolution, parallel, manipulability_mode, collision_detect)

                #Process Trajectory Result to collect metadata
                results.append(trajectory_result)

        print('Post-Processing Collected Data')
        for trajectory_result in results:
            valid = True
            score_average = 1.0
            for res in trajectory_result:
                if res[1] == 0:
                    valid = False
                score_average += res[1]
            score_average = score_average / len(trajectory_result)

            complete_results.append([valid, score_average, trajectory_result])

        return complete_results

    def analyze_manipulability_over_trajectory(self,
                                               point_list,
                                               manipulability_mode=1,
                                               manip_resolution=25,
                                               point_interpolation_dist=-1,
                                               point_interpolation_mode=1,
                                               collision_detect=False):
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
            collision_detect: [Optional Bool] detect collisions (Default False)
        Returns:
            results_list: [(point, score)] results list, one entry per point in trajectory
        """

        #Step One: Flesh out the point list if necessary
        point_list = complete_trajectory(
                point_list, point_interpolation_dist, point_interpolation_mode)

        #Step Two: Determine Manipulability Function
        results = self.analyze_task_space_manipulability(point_list,
                manip_resolution, False, manipulability_mode == 1, collision_detect)

        return results
