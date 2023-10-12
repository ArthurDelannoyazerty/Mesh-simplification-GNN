
from vertex import Vertex
from matrix import Matrix
from face import Face
from pair import PairKey, Pair
from triangle import Triangle
from simplify_funs import *

import numpy as np

import sys

import pdb

class Mesh():
	def __init__(self,triangles):
		self.Triangles = triangles

	def simplify(self, factor,original_num_faces):

		print('\n\nCreating vector -> vertex dict...')
		vectorVertex = {}
		i = 0.0
		nn = float(len(self.Triangles))
		for t in self.Triangles:
			i+=1

			vectorVertex[t.V1] = Vertex(t.V1)
			vectorVertex[t.V2] = Vertex(t.V2)
			vectorVertex[t.V3] = Vertex(t.V3)
			
			amtDone = i /  nn
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))


		print('\n\nCalculating Quadric matrices for each vertex...')
		i = 0.0
		#accumlate quadric matrices for each vertex based on its faces
		for t in self.Triangles:
			i+=1

		
			v1 = vectorVertex[t.V1]
			v2 = vectorVertex[t.V2]
			v3 = vectorVertex[t.V3]
			q1 = t.Quadric(t.V1)
			q2 = t.Quadric(t.V2)
			q3 = t.Quadric(t.V3)

			v1.Quadric = v1.Quadric.Add(q1)
			v2.Quadric = v2.Quadric.Add(q2)
			v3.Quadric = v3.Quadric.Add(q3)

			amtDone = i /  nn
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))
		

		# create faces and map vertex => faces

		vertexFaces = {}
		i = 0.0
		print('\n\nCreating Vertex -> faces dict...')
		for t in self.Triangles:
			i+=1

			v1 = vectorVertex[t.V1]
			v2 = vectorVertex[t.V2]
			v3 = vectorVertex[t.V3]


			f = Face(v1, v2, v3)

			try:
				vertexFaces[v1].append(f)
			except KeyError:
				vertexFaces[v1] = [f]

			try:
				vertexFaces[v2].append(f)
			except KeyError:
				vertexFaces[v2] = [f]

			try:
				vertexFaces[v3].append(f)
			except KeyError:
				vertexFaces[v3] = [f]



			amtDone = i /  nn
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))
		
		
		i = 0.0
		pairs = {}
		print('\n\nCreating pairs...')
		for t in self.Triangles:
			i+=1
			v1 = vectorVertex[t.V1]
			v2 = vectorVertex[t.V2]
			v3 = vectorVertex[t.V3]
			pairs[PairKey(v1, v2)] = Pair(v1, v2)
			pairs[PairKey(v2, v3)] = Pair(v2, v3)
			pairs[PairKey(v3, v1)] = Pair(v3, v1)

			amtDone = i /  nn
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))
		

		pairSharedFaces = {}
		i = 0.0
		n2 = len(pairs.values())
		print('\n\nFinding boundary faces...')
		for p in pairs.values():
			i+=1
			pairSharedFaces[p] = list(set(vertexFaces[p.A]).intersection(set(vertexFaces[p.B])))

			amtDone = i /  n2
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))

		n_one = 0
		n_two = 0
		n2 = len(pairSharedFaces.values())
		i = 0.0
		print('\n\nAdding penalty to boundary pairs...')
		for p,f in zip(pairSharedFaces.keys(), pairSharedFaces.values()):
			i+=1
			if len(f) == 1:
				boundary_q = p.boundary_quadric(f[0])
				p.A.Quadric = p.A.Quadric.Add(boundary_q)
				p.B.Quadric = p.B.Quadric.Add(boundary_q)
				n_one+=1
			else:
				n_two +=1
			amtDone = i /  n2
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))		


		vertexPairs = {}
		i = 0.0
		n = len(pairs)
		heap = {}
		print('\n\nAdding pairs to heap...')
		for p in pairs:
			i+=1
			new_pair = pairs[p]
			heap[new_pair] = new_pair.Error()
			
			try:
				vertexPairs[new_pair.A].append(new_pair)
			except KeyError:
				vertexPairs[new_pair.A] = [new_pair]

			try:
				vertexPairs[new_pair.B].append(new_pair)
			except KeyError:
				vertexPairs[new_pair.B] = [new_pair]
			amtDone = i /  n
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))	



		print('\n\n Deleting Faces...')
		


		iteration = 0
		numFaces = len(self.Triangles)
		print('Orignal Number of Faces: '+ str(original_num_faces))
		#original_num_faces = float(len(self.Triangles))
		target = int(float(original_num_faces) *  factor)
		print('Target Number of Faces: '+str(target))
		#error_cutoff = False
		
		#pdb.set_trace()
		num_repeat_cutoff = 3


		err = np.array(list(heap.values()))
		error_cutoff = np.percentile(err,90)
		del err

		if numFaces > 100000:
			num_repeat_cutoff  = 5
		elif numFaces > 300000:
			num_repeat_cutoff = 7
		elif numFaces > 500000:
			num_repeat_cutoff = 10
		elif numFaces > 1000000:
			num_repeat_cutoff = 15
		elif numFaces > 1500000:
			num_repeat_cutoff = 20

		print('\nStarting Error Cutoff Simplification:')
		for i in range(num_repeat_cutoff):
			print('\nIteration: ' + str(i + 1))
			vertexFaces, vertexPairs, heap, numFaces, target_reached = cutoff_error_simplification(original_num_faces, numFaces, target, heap, vertexFaces, vertexPairs, error_cutoff)
			if target_reached:
				break

		if not target_reached:
			print('\nStarting Minimum Error Simplification')
			vertexFaces = minimum_error_simplification(original_num_faces, numFaces, target, heap, vertexFaces, vertexPairs)

		print('\n\nSimplification Complete')
		distinctFaces = {}
		for faces in vertexFaces.values():
			for f in faces:
				if not f.Removed:
					distinctFaces[f] = True
		
		triangles = [Triangle(f.V1.Vector, f.V2.Vector, f.V3.Vector) for f in distinctFaces if not f.Removed]

		return triangles

