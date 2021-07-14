#!/usr/bin/env python
import sys
from workspace_analyzer import WorkspaceAnalyzer
from robot_link import RobotLink
import matplotlib.pyplot as plt
from faser_math import tm
from faser_utils.disp.disp import disp
from faser_robot_kinematics import loadArmFromURDF
from faser_plotting.Draw.Draw import DrawRectangle, DrawAxes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from workspace_viewer import view_workspace
from alpha_shape import AlphaShape
from workspace_helper_functions import score_point, post_flag
from workspace_helper_functions import load_point_cloud_from_file, sort_cloud
import importlib
import pickle

#The ALPHA_VALUE is the alpha parameter which determines which points are included or
# excluded to create an alpha shape. This is technically an optional parameter,
# however when the alpha shape optimizes this itself, for large clouds of points
# it can take many hours to reach a solution. 1.2 is a convenient constant tested on a variety
# of robot sizes.
ALPHA_VALUE = 1.2

#The TRANSPARENCY_CONSTANT determines the degree of transparency at which 3d objects are shown
# when plotted within matplotlib.
TRANSPARENCY_CONSTANT = .5


done = False

cmd_str_help = 'help'
cmd_str_exit = 'exit'
cmd_str_load_robot = 'loadRobot'
cmd_str_analyze_task = 'analyzeTaskSpace'
cmd_str_analyze_total_workspace_exhaustive = 'exhaustiveMethodTotalWorkspace'
cmd_str_analyze_total_workspace_alpha = 'alphaMethodTotalWorkspace'
cmd_str_analyze_unit_shell_manipulability = 'unitShellManipulability'
cmd_str_analyze_manipulability_object_surface = 'objectSurfaceManipulability'
cmd_str_manipulability_within_volume = 'manipulabilityWithinVolume'
cmd_str_manipulability_over_trajectory = 'manipulabilityOverTrajectory'
cmd_str_view_manipulability_space = 'viewManipulabilitySpace'
cmd_str_schedule_sequence = 'newSequence'
cmd_str_start_sequence = 'startSequence'
cmd_str_view_sequence = 'viewSequence'
cmd_str_load_sequence = 'loadSequence'
cmd_str_save_sequence = 'saveSequence'


valid_commands = [
    cmd_str_help,
    cmd_str_exit,
    cmd_str_load_robot,
    cmd_str_analyze_task,
    cmd_str_analyze_total_workspace_exhaustive,
    cmd_str_analyze_total_workspace_alpha,
    cmd_str_analyze_unit_shell_manipulability,
    cmd_str_analyze_manipulability_object_surface,
    cmd_str_manipulability_within_volume,
    cmd_str_manipulability_over_trajectory,
    cmd_str_view_manipulability_space,
    cmd_str_schedule_sequence,
    cmd_str_start_sequence,
    cmd_str_view_sequence,
    cmd_str_load_sequence,
    cmd_str_save_sequence
]

# General_Flags
cmd_flag_save_results = '-o'
cmd_flag_input_file = '-f'
cmd_flag_num_iters = '-numIterations'
cmd_flag_plot_results = '-plot'
cmd_flag_parallel = '-parallel'
cmd_flag_manip_res = '-manipulationResolution'
cmd_flag_jacobian_manip = '-useJacobian'

# Robot Loading Flags
cmd_flag_from_urdf = '-fromURDF'
cmd_flag_from_file = '-fromPyFile'
cmd_flag_from_directory = '-dir'

# 6DOF Manipulability Flags
cmd_flag_shells_range = '-range'
cmd_flag_shells_num_shells = '-numShells'
cmd_flag_shells_points_per_shell = '-numShellPoints'

# Object Surface Flags
cmd_flag_object_scale = '-scale'
cmd_flag_object_pose = '-pose'
cmd_flag_object_collision_detect = '-checkCollisions'
cmd_flag_object_exempt_ee = '-exemptEE'
cmd_flag_object_bound_volume = '-boundVolume'
cmd_flag_object_min_dist = '-minDist'
cmd_flag_object_approx_collide = '-approxCollisions'

# Within Volume Flags
cmd_flag_volume_grid_res = '-gridResolution'

# Over Trajectory Flags
cmd_flag_trajectory_unit_manip = '-unitSphereManipulability'
cmd_flag_trajectory_interpolation = '-interpolationDist'
cmd_flag_trajectory_arc_interp = '-arcInterpolation'

#Workspace Viewer Flags
cmd_flag_viewer_draw_alpha = '-alphaFile'
cmd_flag_viewer_draw_slices = '-draw3DSlices'


class CommandExecutor:
    """Superclass for workspace command line, to enable use in other applications more easily"""

    def __init(self):
        """Initialize Command Executor"""
        self.analyzer = None
        self.ready = False
        self.done = False
        self.exhaustive_pose_cloud = None
        self.functional_pose_cloud = None
        self.pose_results_6dof_shells = None
        self.pose_results_object_surface = None
        self.pose_results_manipulability_volume = None
        self.pose_results_manipulability_trajectory = None
        self.sequence = []

    def set_sequence_at_launch(self, args):
        """
        Sets a launch sequence so that it can be executed straight from the command line
        Args:
            args: Argument sequence to set up
        """
        if '-sequence' in args:
            sequence_file_name = post_flag('-sequence', args)
            with open(sequence_file_name) as input_file:
                lines = input_file.readlines()
                for line in lines:
                    self.sequence.append(line.strip())
            self.cmd_start_sequence()
            return
        args = sys.argv[1:]
        if '-sequence' in args:
            sequence_file_name = post_flag('-sequence', args)
            with open(sequence_file_name) as input_file:
                lines = input_file.readlines()
                for line in lines:
                    self.sequence.append(line.strip())
            self.cmd_start_sequence()
            return

    def load_robot_from_urdf(self, fname):
        """
        Load a robot from URDF using FASER robotics conventions.
        Args:
            fname: file name of target file
        Returns:
            arm link instance
        """
        arm = loadArmFromURDF(fname)
        link = RobotLink(arm)

        def get_ee():
            return link.robot.getEEPos()

        def do_fk(x):
            return (link.robot.FK(x), True)

        def do_ik(x):
            return link.robot.IK(x, protect=True)

        def get_jt():
            return link.robot.getJointTransforms()
        link.bind_ee(get_ee)
        link.bind_fk(do_fk)
        link.bind_ik(do_ik)
        link.bind_jt(get_jt)
        link.joint_mins = link.robot.joint_mins
        link.joint_maxs = link.robot.joint_maxs
        return link

    def cmd_load_robot(self, cmds):
        """
        loadRobot -from___ file_name.
        Loads a robot for use in the workspace analyzer
        -fromURDF loads from a URDF file using modern_robotics/faser_robotics
        -fromPyFile loads from a user-defined python file specifying the robot
            see the example_robot_module.py file for additional details
        """
        if cmd_flag_from_urdf in cmds:
            file_name = post_flag(cmd_flag_from_urdf, cmds)
            return self.load_robot_from_urdf(file_name)
        if cmd_flag_from_directory in cmds:
            dir_name = post_flag(cmd_flag_from_directory, cmds)
            sys.path.append(dir_name)
        if cmd_flag_from_file in cmds:
            file_name = post_flag(cmd_flag_from_file, cmds)
            file_name_sanitized = file_name.replace('.py','')
            temp_module = importlib.import_module(file_name_sanitized)
            return temp_module.load_arm()

    def cmd_analyze_total_workspace_exhaustive(self, cmds):
        """
        Analyze a robot's total reachability using joint variation (exponential time).
        -numIterations: number of interpolated positions per joint
        -o : save output to file specified
        -plot: displays a basic plot of the results
        """
        num_iterations = 12
        save_output = False
        out_file_name = ''
        plot = False
        if cmd_flag_num_iters in cmds:
            num_iters_str = post_flag(cmd_flag_num_iters, cmds)
            num_iterations = int(num_iters_str)
        if cmd_flag_save_results in cmds:
            out_file_name = post_flag(cmd_flag_save_results, cmds)
            save_output = True
        if cmd_flag_plot_results in cmds:
            plot = True
        pose_cloud = self.analyzer.analyze_total_workspace_exhaustive_point_cloud(num_iterations)
        if plot:
            plt.figure()
            ax = plt.axes(projection='3d')
            alpha = AlphaShape(sort_cloud(pose_cloud), ALPHA_VALUE)
            my_cmap = plt.get_cmap('jet')

            ax.plot_trisurf(*zip(*alpha.verts),
                triangles=alpha.triangle_inds, cmap=my_cmap, alpha=.2, edgecolor='black')
            plt.show()
        self.exhaustive_pose_cloud = pose_cloud
        if save_output:
            with open(out_file_name, 'wb') as fp:
                pickle.dump(pose_cloud, fp)

    def cmd_analyze_total_workspace_functional(self, cmds):
        """
        Analyze a robot's total reachability using single joint variation alpha shape projection.
        Generally runs in linear time
        -numIterations: number of interpolated positions per joint
        -o: save output to file specified
        -plot: displays a basic plot of the results
        """
        num_iterations = 25
        save_output = False
        out_file_name = ''
        plot = False
        if cmd_flag_num_iters in cmds:
            num_iters_str = post_flag(cmd_flag_num_iters, cmds)
            num_iterations = int(num_iters_str)
        if cmd_flag_save_results in cmds:
            out_file_name = post_flag(cmd_flag_save_results, cmds)
            save_output = True
        if cmd_flag_plot_results in cmds:
            plot = True
        pose_cloud = self.analyzer.analyze_total_workspace_functional(num_iterations)
        if plot:
            plt.figure()
            ax = plt.axes(projection='3d')
            alpha = AlphaShape(sort_cloud(pose_cloud), ALPHA_VALUE)
            my_cmap = plt.get_cmap('jet')
            ax.plot_trisurf(*zip(*alpha.verts),
                triangles=alpha.triangle_inds, cmap=my_cmap,alpha =.2, edgecolor='black')
            plt.show()
        self.functional_pose_cloud = pose_cloud
        if save_output:
            with open(out_file_name, 'wb') as fp:
                pickle.dump(pose_cloud, fp)

    def cmd_analyze_manipulability_unit_shells(self, cmds):
        """
        Analyzes workspace manipulability through IK and unit spheres.
        -range: maximum range (meters) of the largest unit sphere from robot origin. Default 4
        -numShells: number of concentric shells from the robot base to range. Default 30
        -numShellPoints: number of points to analyze per shell. Default 1000
        -plot: displays a basic plot of the results
        """
        shells_range = 4
        num_shells = 30
        points_per_shell = 1000
        save_output = False
        out_file_name = ''
        plot = False
        if cmd_flag_shells_range in cmds:
            num_iters_str = post_flag(cmd_flag_shells_range, cmds)
            shells_range = int(num_iters_str)
        if cmd_flag_shells_range in cmds:
            num_iters_str = post_flag(cmd_flag_shells_range, cmds)
            num_shells = int(num_iters_str)
        if cmd_flag_shells_points_per_shell  in cmds:
            num_iters_str = post_flag(cmd_flag_shells_points_per_shell, cmds)
            points_per_shell = int(num_iters_str)
        if cmd_flag_save_results in cmds:
            out_file_name = post_flag(cmd_flag_save_results, cmds)
            save_output = True
        if cmd_flag_plot_results in cmds:
            plot = True
        results = self.analyzer.analyze_6dof_manipulability(
            shells_range, num_shells, points_per_shell)
        if plot:
            plt.figure()
            ax = plt.axes(projection='3d')
            for r in results:
                score = r[1]
                if score < .1:
                    continue
                else:
                    col = score_point(score)
                    DrawRectangle(tm([r[0][0], r[0][1], r[0][2], 0, 0, 0]),
                        [.25] * 3, ax, c=col, a=TRANSPARENCY_CONSTANT)
            plt.show()
        self.pose_results_6dof_shells = results
        if save_output:
            with open(out_file_name, 'wb') as fp:
                pickle.dump(results, fp)

    def cmd_analyze_manipulability_object_surface(self, cmds):
        """
        Analyzes manipulability on the surface of an object
        -f: filename of the stl object to analyze
        -o: Save output to file. Default to unused
        -scale: scale of the object Default 1
        -pose: pose of the object. TAA such as [x,y,z,xr,yr,zr]. M/Rad. No spaces. Default identity
        -manipulationResolution: number of points in manipulation unit sphere. Default 25
        -checkCollisions: make sure no part of the arm intersects the object. Default True
        -exemptEE: exempt the end effector of the robot from collision checking. Default True
        -parallel: run in parallel for increased efficiency. Default False.
        -useJacobian: uses jacobian instead of unit sphere manipulability.
            Incompatible with pose checking
        -plot: displays a basic plot of the results
        -boundVolume: Optional bounds volume
        -minDist: Minimum distance between vertices to consider analyzing
        """
        object_file_name = ''
        object_scale = 1
        object_pose = tm()
        manip_resolution = 25
        collision_detect = True
        exempt_ee = True
        parallel = False
        save_output = False
        plot = False
        use_jacobian = False
        shape = None
        min_dist = -1.0
        out_file_name = ''
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
        if cmd_flag_object_scale in cmds:
            obj_scale_str = post_flag(cmd_flag_object_scale, cmds)
            object_scale = float(obj_scale_str)
        if cmd_flag_object_pose in cmds:
            pose_str = post_flag(cmd_flag_object_pose, cmds)
            object_pose = tm([float(item) for item in pose_str[1:-1].split(',')])
        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))
        if cmd_flag_object_collision_detect in cmds:
            collision_detect = 'true' == post_flag(cmd_flag_object_collision_detect, cmds).lower()
        if cmd_flag_object_exempt_ee in cmds:
            exempt_ee = 'true' == post_flag(cmd_flag_object_exempt_ee, cmds).lower()
        if cmd_flag_parallel in cmds:
            parallel = True
        if cmd_flag_save_results in cmds:
            out_file_name = post_flag(cmd_flag_save_results, cmds)
            save_output = True
        if cmd_flag_plot_results in cmds:
            plot = True
        if cmd_flag_jacobian_manip in cmds:
            use_jacobian = True
        if cmd_flag_object_bound_volume in cmds:
            shape_fname = post_flag(cmd_flag_object_bound_volume, cmds)
            with open(shape_fname, 'rb') as fp:
                shape = AlphaShape(sort_cloud(pickle.load(fp)), ALPHA_VALUE)
        if cmd_flag_object_min_dist in cmds:
            min_dist = float(post_flag(cmd_flag_object_min_dist, cmds))

        results, mesh = self.analyzer.analyze_manipulability_on_object_surface(
            object_file_name, object_scale,
            object_pose, manip_resolution,
            collision_detect, exempt_ee,
            parallel, use_jacobian,
            min_dist, shape)

        if plot:
            plt.figure()
            ax = plt.axes(projection='3d')
            for r in results:
                score = r[1]
                if score < .1:
                    continue
                else:
                    col = score_point(score)
                    DrawRectangle(tm([r[0][0], r[0][1], r[0][2], 0, 0, 0]),
                        [.25] * 3, ax, c=col, a=TRANSPARENCY_CONSTANT)
            ax.add_collection3d(Poly3DCollection(
                mesh.vectors, facecolors='b', edgecolors='r', linewidths=1, alpha=0.1))
            plt.show()
        self.pose_results_object_surface = results

        if save_output:
            with open(out_file_name, 'wb') as fp:
                pickle.dump(results, fp)

    def cmd_analyze_manipulability_within_volume(self, cmds):
        """
        Analyze manipulability within the bounds of a volume.
        -f: filename of shape. Defaults to exhaustive or alpha total workspace if run prior.
            Required if not (.npy)
        -o: output file, if desired
        -gridResolution: desired grid resolution in meters. defaults to .25m
        -manipResolution: desired manipulation resolution (number of unit sphere points) Default 25
        -parallel: Calculate in parallel for increased efficiency. Default false
        -useJacobian: use jacobian optimization instead of unit sphere.
            Faster, but may give less info. Default false
        -plot: displays a basic plot of the results
        """
        shape_fname = ''
        shape = None
        if self.exhaustive_pose_cloud is not None:
            res = input('Would you like to use prior calculated exhaustive pose cloud? Y/N >')
            if res.lower() == 'y':
                shape = AlphaShape(sort_cloud(self.exhaustive_pose_cloud), ALPHA_VALUE)
        if self.functional_pose_cloud is not None:
            res = input('Would you like to use prior calculated functional pose cloud? Y/N >')
            if res.lower() == 'y':
                shape = AlphaShape(sort_cloud(self.functional_pose_cloud), ALPHA_VALUE)

        if shape is None:
            if cmd_flag_input_file in cmds:
                shape_fname = post_flag(cmd_flag_input_file, cmds)
            with open(shape_fname, 'rb') as fp:
                arr = sort_cloud(pickle.load(fp))
            shape = AlphaShape(arr, ALPHA_VALUE)
        if shape is None:
            disp('This function requires an alpha shape cloud as an input\n' +
                'Please either specify a .npy file name or run a pose cloud calculator')
            return

        manip_resolution = 25
        grid_resolution = .25
        parallel = False
        save_output = False
        use_jacobian = False
        plot = False
        out_file_name = ''

        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))
        if cmd_flag_volume_grid_res in cmds:
            grid_resolution = float(post_flag(cmd_flag_volume_grid_res, cmds))
        if cmd_flag_parallel in cmds:
            parallel = True
        if cmd_flag_save_results in cmds:
            out_file_name = post_flag(cmd_flag_save_results, cmds)
            save_output = True
        if cmd_flag_jacobian_manip in cmds:
            use_jacobian = True
        if cmd_flag_plot_results in cmds:
            plot = True
        results = self.analyzer.analyze_manipulability_within_volume(
            shape, grid_resolution, manip_resolution, parallel, use_jacobian)

        if plot:
            plt.figure()
            ax = plt.axes(projection='3d')
            for r in results:
                score = r[1]
                col = score_point(score)
                DrawRectangle(
                    tm([r[0][0], r[0][1], r[0][2], 0, 0, 0]),
                    [grid_resolution]*3, ax, c=col, a=TRANSPARENCY_CONSTANT)
                #disp('Plotting')
            plt.show()
        self.pose_results_manipulability_volume = results
        if save_output:
            with open(out_file_name, 'wb') as fp:
                pickle.dump(results, fp)

    def cmd_analyze_manipulability_over_trajectory(self, cmds):
        """
        Analyzes manipulability over a given trajectory.
        -f: required file name for trajectory in TAA format ([x,y,z,xr,yr,zr]) one per line
        -o: optional output file for saving results
        -unitSphereManipulability: use unit sphere manipulability instead of jacobian manipulability
        -manipResolution: desired manipulation resolution (number of unit sphere points) Default 25
        -interpolationDist: interpolate between points in unit trajectory by a given amount (meters)
        -arcInterpolation: use arc interpolation instead of linear interpolation
        -plot: displays a basic plot of the results
        """
        point_list = []
        manipulability_mode = 1
        manip_resolution = 25
        point_interpolation_dist = -1
        point_interpolation_mode = 1
        save_output = False
        out_file_name = ''
        plot = False

        if cmd_flag_input_file not in cmds:
            disp('A trajectory file is required')
            return
        else:
            point_list = load_point_cloud_from_file(post_flag(cmd_flag_input_file, cmds))

        if cmd_flag_manip_res in cmds:
            manip_resolution = int(post_flag(cmd_flag_manip_res, cmds))

        if cmd_flag_trajectory_unit_manip in cmds:
            manipulability_mode = 2

        if cmd_flag_trajectory_interpolation in cmds:
            point_interpolation_dist = float(post_flag(cmd_flag_trajectory_interpolation, cmds))

        if cmd_flag_trajectory_arc_interp in cmds:
            point_interpolation_mode = 2

        if cmd_flag_save_results in cmds:
            out_file_name = post_flag(cmd_flag_save_results, cmds)
            save_output = True

        if cmd_flag_plot_results in cmds:
            plot = True

        results = self.analyzer.analyze_manipulability_over_trajectory(
            point_list, manipulability_mode, manip_resolution,
            point_interpolation_dist, point_interpolation_mode)

        self.pose_results_manipulability_trajectory = results

        if plot:
            plt.figure()
            ax = plt.axes(projection='3d')
            for r in results:
                DrawAxes(r[0], r[1] / 2, ax)
                ax.scatter3D(r[0][0], r[0][1], r[0][2], c=score_point(r[1]), s=25)
            plt.show()

        if save_output:
            with open(out_file_name, 'wb') as fp:
                pickle.dump(results, fp)

    def cmd_view_manipulability_space(self, cmds):
        """
        Views a previously saved workspace
        -f: Results file to load (.dat)
        -alphafile: alpha shape file to load in (Optional)
        -draw3DSlices: Draw an interactive 3d representation of the alpha slices (Optional)
        """
        object_file_name = ''
        draw_alpha_shape = False
        alpha_shape_file = ''
        draw_slices = False
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
        if cmd_flag_viewer_draw_alpha in cmds:
            alpha_shape_file = post_flag(cmd_flag_viewer_draw_alpha, cmds)
            draw_alpha_shape = True
        if cmd_flag_viewer_draw_slices in cmds:
            draw_slices = True

        view_workspace(object_file_name, draw_alpha_shape, alpha_shape_file, draw_slices)

    def cmd_start_sequence(self):
        """
        Starts or repeats a sequence of commands
        previously defined by the user
        """
        disp('Starting Sequence')
        for cmd in self.sequence:
            self.cmd_parser(cmd)
        disp('Sequence Complete')


class WorkspaceCommandLine(CommandExecutor):
    """
    Provide command line access to workspace_analyzer .
    """

    def __init__(self, args = []):
        super().__init__()
        self.set_sequence_at_launch(args)
        self.main_loop()

    def cmd_schedule_sequence(self):
        """
        Allows for scheduling a sequence of commands
        Presents a prompt to the user
        """
        self.sequence = []
        sequence_done = False
        disp('Welcome to sequence scheduler. To exit, type \'done\'')
        while not sequence_done:
            cmd = input('Sequence Command>')
            if cmd.strip().lower() == 'done':
                disp('Exiting. Use \'startSequence\' to start sequence')
                sequence_done = True
                continue
            confirm = input('Confirm Command?(Y/N)>')
            if confirm.strip().lower() != 'y':
                continue
            self.sequence.append(cmd)

    def cmd_load_sequence(self, cmds):
        """
        Loads a command sequence from file
        -f: file to load from
        """
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
            with open(object_file_name) as input_file:
                lines = input_file.readlines()
                for line in lines:
                    self.sequence.append(line.strip())
        elif len(cmds) > 1:
            object_file_name = cmds[1]
            with open(object_file_name) as input_file:
                lines = input_file.readlines()
                for line in lines:
                    self.sequence.append(line.strip())
        else:
            disp('Please specify a file')

    def cmd_save_sequence(self, cmds):
        """
        Saves a command sequence to file
        -f: file to save to
        """
        if cmd_flag_input_file in cmds:
            object_file_name = post_flag(cmd_flag_input_file, cmds)
            with open(object_file_name, 'w+') as output_file:
                for line in self.sequence:
                    output_file.write(line + '\n')
        else:
            disp('Please specify a file')


    def cmd_view_sequence(self):
        """
        If a user has previously defined a sequence of commands,
        this function will display them for the user to review
        """
        if len(self.sequence) == 0:
            disp('Nothing to show')
        else:
            for i in range(len(self.sequence)):
                disp(str(i+1) + ') ' + self.sequence[i])


    def main_loop(self):
        """Start Main Loop for Command Line Program."""
        self.done = False
        while not self.done:
            cmd = input('Command> ')
            self.cmd_parser(cmd)
            continue
            try:
                cmd = input('Command> ')
                self.cmd_parser(cmd)
            except Exception as e:
                disp('Command failure. Details: ' + str(e))

    def cmd_parser(self, cmd):
        """
        Parse a command string.

        Args:
            cmd: command string passed in
        """
        cmd_no_eq = cmd.replace('=', ' ')
        cmds_parsed = cmd_no_eq.split(' ')
        if not cmds_parsed[0] in valid_commands:
            disp('Unrecognized command. Please try again. You can use \'help\' for help')
            return
        if cmds_parsed[0] == cmd_str_help:
            self.cmd_help(cmds_parsed)
            return
        if cmds_parsed[0] == cmd_str_exit:
            self.done = True
            return False
        elif cmds_parsed[0] == cmd_str_schedule_sequence:
            self.cmd_schedule_sequence()
            return
        elif cmds_parsed[0] == cmd_str_start_sequence:
            self.cmd_start_sequence()
            return
        elif cmds_parsed[0] == cmd_str_view_sequence:
            self.cmd_view_sequence()
            return
        elif cmds_parsed[0] == cmd_str_save_sequence:
            self.cmd_save_sequence(cmds_parsed)
            return
        elif cmds_parsed[0] == cmd_str_load_sequence:
            self.cmd_load_sequence(cmds_parsed)
            return
        elif cmds_parsed[0] == cmd_str_load_robot:
            link = self.cmd_load_robot(cmds_parsed)
            self.analyzer = WorkspaceAnalyzer(link)
            self.ready = True
            return
        if not self.ready:
            disp('Please load a robot first')
            return
        if cmds_parsed[0] == cmd_str_analyze_total_workspace_exhaustive:
            self.cmd_analyze_total_workspace_exhaustive(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_analyze_total_workspace_alpha:
            self.cmd_analyze_total_workspace_functional(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_analyze_unit_shell_manipulability:
            self.cmd_analyze_manipulability_unit_shells(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_analyze_manipulability_object_surface:
            self.cmd_analyze_manipulability_object_surface(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_manipulability_within_volume:
            self.cmd_analyze_manipulability_within_volume(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_manipulability_over_trajectory:
            self.cmd_analyze_manipulability_over_trajectory(cmds_parsed)
        elif cmds_parsed[0] == cmd_str_view_manipulability_space:
            self.cmd_view_manipulability_space(cmds_parsed)

    def cmd_help(self, cmds):
        """
        Assist the user in function understanding.

        Args:
            cmds: command passed through, for specific help if needed
        """
        if len(cmds) > 1:
            if cmds[1] == cmd_str_exit:
                disp('Exits the program')
            elif cmds[1] == cmd_str_load_robot:
                disp(self.cmd_load_robot.__doc__)
            elif cmds[1] == cmd_str_analyze_total_workspace_exhaustive:
                disp(self.cmd_analyze_total_workspace_exhaustive.__doc__)
            elif cmds[1] == cmd_str_analyze_total_workspace_alpha:
                disp(self.cmd_analyze_total_workspace_functional.__doc__)
            elif cmds[1] == cmd_str_analyze_unit_shell_manipulability:
                disp(self.cmd_analyze_manipulability_unit_shells.__doc__)
            elif cmds[1] == cmd_str_analyze_manipulability_object_surface:
                disp(self.cmd_analyze_manipulability_object_surface.__doc__)
            elif cmds[1] == cmd_str_manipulability_within_volume:
                disp(self.cmd_analyze_manipulability_within_volume.__doc__)
            elif cmds[1] == cmd_str_manipulability_over_trajectory:
                disp(self.cmd_analyze_manipulability_over_trajectory.__doc__)
            elif cmds[1] == cmd_str_schedule_sequence:
                disp(self.cmd_schedule_sequence.__doc__)
            elif cmds[1] == cmd_str_start_sequence:
                disp(self.cmd_start_sequence.__doc__)
            elif cmds[1] == cmd_str_view_sequence:
                disp(self.cmd_view_sequence.__doc__)
            elif cmds[1] == cmd_str_load_sequence:
                disp(self.cmd_load_sequence.__doc__)
            elif cmds[1] == cmd_str_save_sequence:
                disp(self.cmd_save_sequence.__doc__)
            elif cmds[1] == cmd_str_view_manipulability_space:
                disp(self.cmd_view_manipulability_space.__doc__)
        else:
            disp('\nhelp [cmd] for details\nValid Commands:')
            for command in valid_commands:
                disp('\t'+command)
            disp('\n')


if __name__ == '__main__':
    command_line = WorkspaceCommandLine()
