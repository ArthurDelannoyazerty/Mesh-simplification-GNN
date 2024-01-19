# simplification2/__init__.py

# Import each module you want to be accessible when 'from simplification2 import *' is used
from .mesh import Mesh
from .simplify_meshh import simplification2
from .triangle import Triangle
from .matrix import Matrix
from .vertex import Vertex
from .face import Face
from .pair import PairKey, Pair
from .vector import Vector
from .simplify_funs import *


# ... import other modules as needed

# Define what should be imported with the asterisk (*) import

_all_ = ['Mesh', 'simplification2', 'Triangle', 'Matrix', 'Vertex', 'Face', 'PairKey', 'Pair', 'Vector', 'minimum_error_simplification', 'simplification2']