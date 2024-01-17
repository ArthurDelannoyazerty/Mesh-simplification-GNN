from math import sqrt

class Vector():
	def __init__(self, x, y, z):
		self.X = x
		self.Y = y
		self.Z = z

	
	def __eq__(self, other):
		return (self.X == other.X) and (self.Y == other.Y) and (self.Z == other.Z) 

	def __hash__(self):
		return hash((self.X, self.Y, self.Z))

	def Less(self, b):
		if self.X != b.X:
			return self.X < b.X
		
		if self.Y != b.Y:
			return self.Y < b.Y
		
		return self.Z < b.Z

	def Length(self):
		return sqrt(self.X*self.X + self.Y*self.Y + self.Z*self.Z)

	def Dot(self, b):
		return self.X*b.X + self.Y*b.Y + self.Z*b.Z

	def Cross(self,b):
		x = self.Y*b.Z - self.Z*b.Y
		y = self.Z*b.X - self.X*b.Z
		z = self.X*b.Y - self.Y*b.X
		return Vector(x, y, z)

	def Normalize(self):
		d = self.Length()
		if d < 1E-5:
			return Vector(0.,0.,0.)
		return Vector(self.X / d, self.Y / d, self.Z / d)

	def Sub(self, b):
		return Vector(self.X - b.X, self.Y - b.Y, self.Z - b.Z)

	def Add(self, b):
		return Vector(self.X + b.X, self.Y + b.Y, self.Z + b.Z)

	def MulScalar(self,b):
		return Vector(self.X * b, self.Y * b, self.Z * b)


