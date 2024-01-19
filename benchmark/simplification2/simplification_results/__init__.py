# simplification2/__init__.py

# Import each module you want to be accessible when 'from simplification2 import *' is used
from mesh import Mesh
from simplify_mesh import simplify_mesh
from triangle import Triangle
from vertex import Vertex
from matrix import Matrix
from face import Face
from pair import PairKey, Pair
from simplify_funs import *

# ... import other modules as needed

# Define what should be imported with the asterisk (*) import
__all__ = ['Mesh', 'simplify_mesh', 'Triangle', 'Vertex', 'Matrix', 'Face', 'PairKey', 'Pair', 'simplify_funs']