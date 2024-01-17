
from matrix import Matrix

class Triangle():
	def __init__(self, v1, v2, v3):
		self.V1 = v1  # Class Vector
		self.V2 = v2
		self.V3 = v3


	def Quadric(self, vert):
		n = self.Normal()
		x, y, z = vert.X, vert.Y, vert.Z
		a, b, c = n.X, n.Y, n.Z
		d = -a*x - b*y - c*z
		return Matrix(
			a * a, a * b, a * c, a * d, \
			a * b, b * b, b * c, b * d, \
			a * c, b * c, c * c, c * d, \
			a * d, b * d, c * d, d * d)

	def Normal(self):
		e1 = self.V2.Sub(self.V1)
		e2 = self.V3.Sub(self.V1)

		return e1.Cross(e2).Normalize()


