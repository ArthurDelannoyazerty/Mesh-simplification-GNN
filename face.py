from vector import Vector

class Face():
	def __init__(self, v1, v2, v3):
		


		self.V1 = v1  # Class Vertex
		self.V2 = v2
		self.V3 = v3
		self.Removed = False

	def Degenerate(self):

		same_vector = self.V1.Vector == self.V2.Vector or self.V1.Vector == self.V3.Vector or self.V2.Vector == self.V3.Vector


		return same_vector or self.is_one_dimensional()


	def is_one_dimensional(self):
		e1 = self.V2.Vector.Sub(self.V1.Vector)
		e2 = self.V3.Vector.Sub(self.V1.Vector)	
	
		return e1.Cross(e2).Normalize() == Vector(0.,0.,0.)

	def Normal(self):
		e1 = self.V2.Vector.Sub(self.V1.Vector)
		e2 = self.V3.Vector.Sub(self.V1.Vector)

		return e1.Cross(e2).Normalize()




