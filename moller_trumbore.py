import numpy as np
import workspace_constants
from alpha_shape import AlphaShape
from basic_robotics.utilities.disp import disp, progressBar

MOLLER_BIAS = np.pi/1000 #Arbitrary Small Amount
MOLLER_RAY = np.array([MOLLER_BIAS, 2 * MOLLER_BIAS, 1 - 3 * MOLLER_BIAS])

def moller_trumbore_ray_intersection(origin_point,
                                     triangle,
                                     ray_dir=MOLLER_RAY):
    """
    Execute Moller-Trumbore triangle/Ray Intersection Algorithm on a single point
    Args:
        origin_point: the point at which the ray originates
        triangle: List of three points in 3D space denoting a triangle
        ray_dir: Optional List of three points denoting a unit vector/ray
    Returns:
        Boolean for whether or not the ray intersects the triangle
    """
    # Algorithm sourced from
    # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    vec_0 = triangle[0]
    vec_1 = triangle[1]
    vec_2 = triangle[2]

    ray = np.array(ray_dir)
    edge_1 = vec_1 - vec_0
    edge_2 = vec_2 - vec_0

    h = np.cross(ray, edge_2)
    a = np.dot(edge_1, h)
    if -workspace_constants.EPSILON < a < workspace_constants.EPSILON:
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
    if t <= workspace_constants.EPSILON:
        return False

    return True


def moller_trumbore_ray_intersection_array(points,
                                           triangle,
                                           ray=MOLLER_RAY):
    """
    Executes Moller-Trumbore triangle/Ray Intersection Algorithm on a group
    of points simultaneously, utilizing numpy array expressions.
    Args:
        points: A list of points to test on the algorithm
        triangle: List of three points in 3D space denoting a triangle
        ray: Optional array of three points denoting a unit vector/ray
    Returns:
        Numpy array of booleans for which corresponding points intersect the
            triangle
    """
    # Determine the vectors that make up the triangle to test
    vec_0 = triangle[0]
    vec_1 = triangle[1]
    vec_2 = triangle[2]
    alen = len(points)

    if alen == 3:  # If there is only one point, do the regular ray intersection algorithm
        return moller_trumbore_ray_intersection(points, triangle)

    edge_1 = vec_1 - vec_0
    edge_2 = vec_2 - vec_0
    h = np.cross(ray, edge_2)
    a = np.dot(edge_1, h)

    if -workspace_constants.EPSILON < a < workspace_constants.EPSILON:
        return np.array([False] * alen)  # Ray is parallel to triangle so intersects

    # Here it becomes Matrix Ops
    f = 1.0 / a
    res = np.array([True] * alen)
    s = points - vec_0
    u = f * np.dot(s, h)

    res[np.where(u < 0.0)] = False
    res[np.where(u > 1.0)] = False

    q = np.cross(s, edge_1)
    v = f * (np.dot(q, ray))
    res[np.where(v < 0.0)] = False
    res[np.where((u + v) > 1.0)] = False
    t = f * np.dot(q, edge_2)

    res[np.where(t <= workspace_constants.EPSILON)] = False

    return res


def chunk_moller_trumbore(test_points, triangles, result_array):
    """
    Perform a chunk of moller_trumbore, useful for parallel processing.

    Args:
        test_points: points to process [nx3]
        triangles: list/array of triangles to test with moller_trumbore
        result_array: [nx1] vector of ints used to count triangle intersections
    Returns:
        result_array
    """
    for i in range(np.shape(triangles)[0]):
        res = moller_trumbore_ray_intersection_array(test_points,
                                                     triangles[i, :])
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

    num_points = len(points)
    if num_points < 1:
        # Attempted to analyze empty point set
        return []

    inside_alpha = []
    if check_bounds:
        # Before we start moller_trumbore, eliminate any points that are outside of the known bounds
        i = 0
        if num_points == 1:
            if (points[0][0] < bounds[0][0] or points[0][0] > bounds[0][1]
                    or points[0][1] < bounds[1][0] or points[0][1] > bounds[1][1]
                    or points[0][2] < bounds[2][0] or points[0][2] > bounds[2][1]):
                # If the point is outside of the bounds, return the empty set.
                return []
            inside_alpha.append(points[0])
        else:
            for point in points:
                progressBar(i, num_points - 1, prefix='Checking Bounds             ')
                i += 1
                if (point[0] < bounds[0][0] or point[0] > bounds[0][1]
                        or point[1] < bounds[1][0] or point[1] > bounds[1][1]
                        or point[2] < bounds[2][0] or point[2] > bounds[2][1]):
                    # Erasure by continuing
                    continue
                else:
                    inside_alpha.append(point)
        inside_alpha = np.array(inside_alpha)
    else:
        # If didn't opt to check bounds, the target list will be that which was originally input
        inside_alpha = points
    if len(inside_alpha) == 0:
        return []
    triangles = shape.triangles
    points = inside_alpha
    num_tri = len(triangles)
    inside = []
    i = 0
    num_intersections = np.zeros(len(points))
    #Individually process each triangle through Moller trumbore, as one would trial line segments
    #   in the point-in-polygon problem.
    for tri in triangles:
        progressBar(i, num_tri - 1, prefix='Running Moller Trumbore     ')
        i += 1
        res = moller_trumbore_ray_intersection_array(
            points, tri)
        num_intersections[np.where(res)] += 1

    if len(np.where(isodd(num_intersections))[0]) == 0:
        return []  # IF nothing is odd, then there's nothing in the alpha shape

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
    new_shape = AlphaShape(new_points, workspace_constants.ALPHA_VALUE)
    return new_shape
