from vector import Vector

class Matrix():
	def __init__(self, x00, x01, x02, x03, x10, x11, x12, x13, x20, x21, x22, x23, x30, x31, x32, x33):
		self.x00 = x00
		self.x01 = x01
		self.x02 = x02
		self.x03 = x03
		self.x10 = x10
		self.x11 = x11
		self.x12 = x12
		self.x13 = x13
		self.x20 = x20
		self.x21 = x21
		self.x22 = x22
		self.x23 = x23
		self.x30 = x30
		self.x31 = x31
		self.x32 = x32
		self.x33 = x33

	def QuadricError(self, v):
		return (v.X*self.x00*v.X + v.Y*self.x10*v.X + v.Z*self.x20*v.X + self.x30*v.X + \
		v.X*self.x01*v.Y + v.Y*self.x11*v.Y + v.Z*self.x21*v.Y + self.x31*v.Y + \
		v.X*self.x02*v.Z + v.Y*self.x12*v.Z + v.Z*self.x22*v.Z + self.x32*v.Z + \
		v.X*self.x03 + v.Y*self.x13 + v.Z*self.x23 + self.x33)

	def der_quadric_matrix(self):
		return Matrix(
		self.x00, self.x01, self.x02, self.x03, \
		self.x10, self.x11, self.x12, self.x13, \
		self.x20, self.x21, self.x22, self.x23, \
		0., 0., 0., 1.)

	def QuadricVector(self):
		b = Matrix(
		self.x00, self.x01, self.x02, self.x03, \
		self.x10, self.x11, self.x12, self.x13, \
		self.x20, self.x21, self.x22, self.x23, \
		0, 0, 0, 1,)
	
		return b.Inverse().MulPosition(Vector(0.,0.,0.))

	def Add(self, b):
		return Matrix(
		self.x00 + b.x00, self.x10 + b.x10, self.x20 + b.x20, self.x30 + b.x30, \
		self.x01 + b.x01, self.x11 + b.x11, self.x21 + b.x21, self.x31 + b.x31, \
		self.x02 + b.x02, self.x12 + b.x12, self.x22 + b.x22, self.x32 + b.x32, \
		self.x03 + b.x03, self.x13 + b.x13, self.x23 + b.x23, self.x33 + b.x33, \
	)
	
	def multiply_scalar(self,s):
		return Matrix(self.x00*s, self.x01*s, self.x02*s, self.x03*s, self.x10*s, self.x11*s, self.x12*s, self.x13*s, self.x20*s, self.x21*s, self.x22*s, self.x23*s, self.x30*s, self.x31*s, self.x32*s, self.x33*s)

	def MulPosition(self, b):
		x = self.x00*b.X + self.x01*b.Y + self.x02*b.Z + self.x03
		y = self.x10*b.X + self.x11*b.Y + self.x12*b.Z + self.x13
		z = self.x20*b.X + self.x21*b.Y + self.x22*b.Z + self.x23
		return Vector(x, y, z)

	
	def Determinant(self):
		return (self.x00*self.x11*self.x22*self.x33 - self.x00*self.x11*self.x23*self.x32 + \
		self.x00*self.x12*self.x23*self.x31 - self.x00*self.x12*self.x21*self.x33 + \
		self.x00*self.x13*self.x21*self.x32 - self.x00*self.x13*self.x22*self.x31 - \
		self.x01*self.x12*self.x23*self.x30 + self.x01*self.x12*self.x20*self.x33 - \
		self.x01*self.x13*self.x20*self.x32 + self.x01*self.x13*self.x22*self.x30 - \
		self.x01*self.x10*self.x22*self.x33 + self.x01*self.x10*self.x23*self.x32 + \
		self.x02*self.x13*self.x20*self.x31 - self.x02*self.x13*self.x21*self.x30 + \
		self.x02*self.x10*self.x21*self.x33 - self.x02*self.x10*self.x23*self.x31 + \
		self.x02*self.x11*self.x23*self.x30 - self.x02*self.x11*self.x20*self.x33 - \
		self.x03*self.x10*self.x21*self.x32 + self.x03*self.x10*self.x22*self.x31 - \
		self.x03*self.x11*self.x22*self.x30 + self.x03*self.x11*self.x20*self.x32 - \
		self.x03*self.x12*self.x20*self.x31 + self.x03*self.x12*self.x21*self.x30)

	def Inverse(self):

		r = 1 / self.Determinant()
		x00 = (self.x12*self.x23*self.x31 - self.x13*self.x22*self.x31 + self.x13*self.x21*self.x32 - self.x11*self.x23*self.x32 - self.x12*self.x21*self.x33 + self.x11*self.x22*self.x33) * r
		x01 = (self.x03*self.x22*self.x31 - self.x02*self.x23*self.x31 - self.x03*self.x21*self.x32 + self.x01*self.x23*self.x32 + self.x02*self.x21*self.x33 - self.x01*self.x22*self.x33) * r
		x02 = (self.x02*self.x13*self.x31 - self.x03*self.x12*self.x31 + self.x03*self.x11*self.x32 - self.x01*self.x13*self.x32 - self.x02*self.x11*self.x33 + self.x01*self.x12*self.x33) * r
		x03 = (self.x03*self.x12*self.x21 - self.x02*self.x13*self.x21 - self.x03*self.x11*self.x22 + self.x01*self.x13*self.x22 + self.x02*self.x11*self.x23 - self.x01*self.x12*self.x23) * r
		x10 = (self.x13*self.x22*self.x30 - self.x12*self.x23*self.x30 - self.x13*self.x20*self.x32 + self.x10*self.x23*self.x32 + self.x12*self.x20*self.x33 - self.x10*self.x22*self.x33) * r
		x11 = (self.x02*self.x23*self.x30 - self.x03*self.x22*self.x30 + self.x03*self.x20*self.x32 - self.x00*self.x23*self.x32 - self.x02*self.x20*self.x33 + self.x00*self.x22*self.x33) * r
		x12 = (self.x03*self.x12*self.x30 - self.x02*self.x13*self.x30 - self.x03*self.x10*self.x32 + self.x00*self.x13*self.x32 + self.x02*self.x10*self.x33 - self.x00*self.x12*self.x33) * r
		x13 = (self.x02*self.x13*self.x20 - self.x03*self.x12*self.x20 + self.x03*self.x10*self.x22 - self.x00*self.x13*self.x22 - self.x02*self.x10*self.x23 + self.x00*self.x12*self.x23) * r
		x20 = (self.x11*self.x23*self.x30 - self.x13*self.x21*self.x30 + self.x13*self.x20*self.x31 - self.x10*self.x23*self.x31 - self.x11*self.x20*self.x33 + self.x10*self.x21*self.x33) * r
		x21 = (self.x03*self.x21*self.x30 - self.x01*self.x23*self.x30 - self.x03*self.x20*self.x31 + self.x00*self.x23*self.x31 + self.x01*self.x20*self.x33 - self.x00*self.x21*self.x33) * r
		x22 = (self.x01*self.x13*self.x30 - self.x03*self.x11*self.x30 + self.x03*self.x10*self.x31 - self.x00*self.x13*self.x31 - self.x01*self.x10*self.x33 + self.x00*self.x11*self.x33) * r
		x23 = (self.x03*self.x11*self.x20 - self.x01*self.x13*self.x20 - self.x03*self.x10*self.x21 + self.x00*self.x13*self.x21 + self.x01*self.x10*self.x23 - self.x00*self.x11*self.x23) * r
		x30 = (self.x12*self.x21*self.x30 - self.x11*self.x22*self.x30 - self.x12*self.x20*self.x31 + self.x10*self.x22*self.x31 + self.x11*self.x20*self.x32 - self.x10*self.x21*self.x32) * r
		x31 = (self.x01*self.x22*self.x30 - self.x02*self.x21*self.x30 + self.x02*self.x20*self.x31 - self.x00*self.x22*self.x31 - self.x01*self.x20*self.x32 + self.x00*self.x21*self.x32) * r
		x32 = (self.x02*self.x11*self.x30 - self.x01*self.x12*self.x30 - self.x02*self.x10*self.x31 + self.x00*self.x12*self.x31 + self.x01*self.x10*self.x32 - self.x00*self.x11*self.x32) * r
		x33 = (self.x01*self.x12*self.x20 - self.x02*self.x11*self.x20 + self.x02*self.x10*self.x21 - self.x00*self.x12*self.x21 - self.x01*self.x10*self.x22 + self.x00*self.x11*self.x22) * r
		return Matrix(x00, x01, x02, x03, x10, x11, x12, x13, x20, x21, x22, x23, x30, x31, x32, x33)

