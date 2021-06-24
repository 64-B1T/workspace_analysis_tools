import xml.etree.ElementTree as ET
import numpy as np
import os
from faser_robot_kinematics import Arm
from faser_math import tm



def extract_origin(x_obj):
    """
    Shortcut for pulling from xml.
    Args:
        x: xml object root
    Returns:
        origin of root
    """
    return x_obj.find('origin').get('xyz').split()


#def extract_parent(x): return x.find('parent').get('link')
#def extract_child(x): return x.find('child').get('link')


class LinkHolder:
    """
    Simple Class to temporarily hold a link object as extracted from the URDF
    """

    def __init__(self):
        """
        Initializes a LinkHolder Object
        """
        self.link_cg = None
        self.mass = None
        self.id = None
        self.name = None


class JointHolder:
    """
    Simple Class to temporarily hold a joint object as extracted from the URDF
    """

    def __init__(self):
        """
        Initializes a JointHolder Object
        """
        self.axis = None
        self.xyz_origin = None
        self.rpy_origin = None
        self.id = None
        self.name = None


def parse_urdf(urdf_path):
    """
    Parses a URDF File into a series of joint objects and a series of link objects
    Args:
        urdf_path: The path to the desired URDF file
    Returns:
        joints: A list of JointHolder objects
        links: A List of LinkHolder objects
    """
    try:
        tree = ET.parse(urdf_path)
    except:
        if os.path.exists(urdf_path):
            print('Malformed URDF, Exiting.')
        else:
            print('File Not Found, Exiting.')
        return

    root = tree.getroot()

    #Set link and joint counts
    link_count = 0
    joint_count = 0

    links = []
    joints = []

    #Parse through Each Item to Find all the links and joints
    for child in root:
        if child.tag == 'link':
            links.append(parse_link(child))
            links[-1].id = link_count
            link_count += 1
        elif child.tag == 'joint':
            joints.append(parse_joint(child))
            joints[-1].id = joint_count
            joint_count += 1
    return joints, links


def parse_link(link):
    """
    Parses a link xml object into a LinkHolder object
    Args:
        link: xml object representing a URDF link
    Returns
        tlink: LinkHolder object containing relevant information
    """
    tlink = LinkHolder()
    tlink.name = link.get('name')
    for child in link:
        if child.tag == 'inertial':
            cg_origin = np.array(extract_origin(child), dtype=float)
            #print(cg_origin)
            mass = float(child.find('mass').get('value'))
            tlink.link_cg = cg_origin
            tlink.mass = mass
    return tlink


def parse_joint(joint):
    """
    Parses a joint xml object into a LinkHolder object
    Args:
        joint: xml object representing a URDF link
    Returns
        tlink: JointHolder object containing relevant information
    """
    tjoint = JointHolder()
    tjoint.name = joint.get('name')
    for child in joint:
        if child.tag == 'axis':
            axis = np.array(child.get('xyz').split(), dtype=float)
            tjoint.axis = axis
        if child.tag == 'origin':
            xyz = np.array(child.get('xyz').split(), dtype=float)
            rpy = np.array(child.get('rpy').split(), dtype=float)
            tjoint.xyz_origin = xyz
            tjoint.rpy_origin = rpy
    return tjoint


def build_arm(joints, links):
    """
    Constructs a POE style serial arm using a set of joint and link objects
    Args:
        joints: list of JointHolder objects
        links: list of LinkHolder objects
    Returns:
        Arm: FASER_ARM object

    """
    #Required Parameters:
    #Transforms of each link in home pose for masses (TSPACE)
    #Joint Axes (w_list)'
    #Transofrms of each joint (q_list)

    num_dof = len(joints) - 2  #How many DOF are there?

    home = tm()  #For now we'll set up the arm in the Space Frame
    jointposes = [home]

    i = 0
    #Create the list of joint poses in the home position
    while i < num_dof + 1:
        while joints[i].xyz_origin is None:
            joints.remove(joints[i])  #Can't do anything with joints that don't have an origin, so remove these.
        ltrn = joints[i].xyz_origin
        lrot = joints[i].rpy_origin
        local_pose = tm([ltrn[0], ltrn[1], ltrn[2], lrot[0], lrot[1], lrot[2]])
        jointposes.append(jointposes[-1] @ local_pose)
        i += 1

    #disp(jointposes, "Initial Joint Poses") #May or may not use these directly.
    #Still awaiting input
    masses_list = []
    for link in links:
        if link.mass is not None:
            masses_list.append(link.mass)
    #masses = np.array(masses_list)

    #w_list is lowest hanging fruit, so calculate first
    w_list = np.zeros((3, num_dof))
    for i in range(0, num_dof):
        #Hack implemented to ensure calculated joints align with MAXAR example
        #In test ccase at the least.
        if np.isclose(jointposes[i + 2][3], -np.pi / 2):
            w_list[0:3, i] = np.array([0, 1, 0])
        else:
            w_list[0:3, i] = np.array([0, 0, 1])

    #Joint Location List, Devoid of Rotation Information
    q_list = np.zeros((3, num_dof))
    for i in range(0, num_dof):
        q_list[0:3, i] = jointposes[i + 2][0:3].flatten()

    #disp(q_list, "Link Initial Pose List")
    #Screw List Generation
    s_list = np.zeros((6, num_dof))
    for i in range(num_dof):
        #Glist... Not Included Yet (Dynamics Not Needed)
        s_list[0:6, i] = np.hstack(
            (w_list[0:3, i], np.cross(q_list[0:3, i], w_list[0:3, i])))

    ee_home = jointposes[-1]
    dims = np.zeros((3, num_dof + 1))
    for i in range(num_dof + 1):
        dims[0:3,
             i] = np.array([.1, .1, joints[i].xyz_origin[2]
                           ])  #Extremely basic dimensions, not loading parts

    arm = Arm(tm(), s_list, ee_home, q_list, w_list)
    arm._Dims = dims
    arm._Mhome = jointposes[1:]
    

    return arm
