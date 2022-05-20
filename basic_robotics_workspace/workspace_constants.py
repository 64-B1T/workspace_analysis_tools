EPSILON = .000001  # Deviation Acceptable for Moller Trumbore
RAY_UNIT = 10  # Unit Vector Constant in Moller Trumbore. Arbitrary
MAX_DIST = 10  # Maximum distance from robot base to discount points in object surface
UNIQUE_DECIMALS = 2  # Decimal places to filter out for uniqueness in Alpha Shapes

# Number of DOF from the End Effector assumed to be capable of creating a 3d shape
# Through varying of the last n DOF and tracking End Effector coordinates.
# Insufficient DOF OFFSET can result in the alpha shape reachability solver failing.
DOF_OFFSET = 3

#The ALPHA_VALUE is the alpha parameter which determines which points are included or
# excluded to create an alpha shape. This is technically an optional parameter,
# however when the alpha shape optimizes this itself, for large clouds of points
# it can take many hours to reach a solution. 1.2 is a convenient constant tested on a variety
# of robot sizes.
ALPHA_VALUE = 1.2

#Constant values DOF_OFFSET, ALPHA_VALUE, UNIQUE_DECIMALS, and MAX_DIST are used to initialize
#Class variables of the same function, to allow tuning *if required*
#They are reproduced here at the top of the file as constants for visibility and convenience


#The TRANSPARENCY_CONSTANT determines the degree of transparency at which 3d objects are shown
# when plotted within matplotlib.
TRANSPARENCY_CONSTANT = .5
