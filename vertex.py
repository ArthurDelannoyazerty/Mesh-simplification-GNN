

from matrix import Matrix
from vector import Vector

class Vertex:
	def __init__(self,v,Q = Matrix(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)):
		self.Vector = v
		self.Quadric = Q

	def __eq__(self, other):
		return (self.Vector.__eq__(other.Vector))

	def __hash__(self):
		return hash((self.Vector.X, self.Vector.Y, self.Vector.Z))