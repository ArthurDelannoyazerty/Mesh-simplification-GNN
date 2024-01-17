# simplify functions

from vertex import Vertex
from matrix import Matrix
from face import Face
from pair import PairKey, Pair
import sys
import operator
import copy

def minimum_error_simplification(original_num_faces, numFaces, target, heap, vertexFaces, vertexPairs):
	iteration = 0
	while numFaces > target:

		iteration +=1
		#print('\nIteration '+ str(iteration))
		pct = ((original_num_faces - target) - (numFaces - target)) / (original_num_faces - target) 
		sys.stdout.write("\rProgress: [{0:50s}] {1:.3f}%".format('#' * int(pct * 50), pct*100))	

		#print('% Complete: ' + str(pct)+ '%')


		try:
			p = min(heap.items(), key = operator.itemgetter(1))[0]
		except ValueError:
			break

		
		del heap[p]
		if p.Removed:
			continue

		p.Removed = True


		distinctFaces = set()
		for f in vertexFaces[p.A]:
			if not f.Removed:
				distinctFaces.add(f)

		for f in vertexFaces[p.B]:
			if not f.Removed:
				distinctFaces.add(f)


		distinctPairs = set()
		for q in vertexPairs[p.A]:
			if not q.Removed:
				distinctPairs.add(q)


		for q in vertexPairs[p.B]:
			if not q.Removed:
				distinctPairs.add(q)


		v = Vertex(p.Vector(), p.Quadric())

		

		newFaces = []
		new_face_verts = set()
		valid = True
		for f in distinctFaces:
			v1,v2,v3 = f.V1, f.V2, f.V3
			

			if v1 == p.A or v1 == p.B:
				v1 = v
			
			if v2 == p.A or v2 == p.B:
				v2 = v
			
			if v3 == p.A or v3 == p.B:
				v3 = v
			
			face = Face(v1, v2, v3)
			if face.Degenerate():
				continue
			
			if face.Normal().Dot(f.Normal()) < 1e-3:
				
				valid = False
				break
			
		
			newFaces.append(face)
			new_face_verts.add(face.V1)
			new_face_verts.add(face.V2)
			new_face_verts.add(face.V3)

		
		
		if not valid:
			continue



		del vertexFaces[p.A]

		del vertexFaces[p.B]
	


		for f in distinctFaces:
			f.Removed = True
			numFaces -= 1

		for f in newFaces:
	
			numFaces +=1
			try:
				vertexFaces[f.V1].append(f)
			except KeyError:
				vertexFaces[f.V1] = [f]
			
			try:
				vertexFaces[f.V2].append(f)
			except KeyError:
				vertexFaces[f.V2] = [f]


			try:
				vertexFaces[f.V3].append(f)
			except KeyError:
				vertexFaces[f.V3] = [f]


		del vertexPairs[p.A]


		del vertexPairs[p.B]


		seen = set()


		for q in distinctPairs:
		
			q.Removed = True
			try:
				del heap[q]
			except KeyError:
				pass

			a,b = q.A, q.B


			if a == p.A or a == p.B:
				a = v

			if b == p.A or b == p.B:
				b = v

			if b == v:
				a,b = b,a


			if b.Vector in seen:
					continue
			if a not in new_face_verts or b not in new_face_verts:
				continue

			if a == b:
				continue

			seen.add(b.Vector)
			q = Pair(a,b)


			heap[q]=q.Error()

			try:
				vertexPairs[a].append(q)
			except KeyError:
				vertexPairs[a] = [q]
			try:
				vertexPairs[b].append(q)
			except KeyError:
				vertexPairs[b] = [q]

	return vertexFaces



# simplify functions


def cutoff_error_simplification(original_num_faces, numFaces, target, heap, vertexFaces, vertexPairs,error_cutoff):
	iteration = 0
	heap_copy = copy.deepcopy(heap)
	target_reached = False
	for p, pair_error in heap.items():

		iteration +=1
		#print('\nIteration '+ str(iteration))
		pct = ((original_num_faces - target) - (numFaces - target)) / (original_num_faces - target) 
		sys.stdout.write("\rProgress: [{0:50s}] {1:.3f}%".format('#' * int(pct * 50), pct*100))	

		#print('% Complete: ' + str(pct)+ '%')

		if numFaces <= target:
			target_reached = True
			break 

		if pair_error > error_cutoff:
			continue

		try:
			del heap_copy[p]
		except KeyError:
			continue

		if p.Removed:
			continue

		p.Removed = True


		distinctFaces = set()
		for f in vertexFaces[p.A]:
			if not f.Removed:
				distinctFaces.add(f)

		for f in vertexFaces[p.B]:
			if not f.Removed:
				distinctFaces.add(f)


		distinctPairs = set()
		for q in vertexPairs[p.A]:
			if not q.Removed:
				distinctPairs.add(q)


		for q in vertexPairs[p.B]:
			if not q.Removed:
				distinctPairs.add(q)


		v = Vertex(p.Vector(), p.Quadric())

		

		newFaces = []
		new_face_verts = set()
		valid = True
		for f in distinctFaces:
			v1,v2,v3 = f.V1, f.V2, f.V3
			

			if v1 == p.A or v1 == p.B:
				v1 = v
			
			if v2 == p.A or v2 == p.B:
				v2 = v
			
			if v3 == p.A or v3 == p.B:
				v3 = v
			
			face = Face(v1, v2, v3)
			if face.Degenerate():
				continue
			
			if face.Normal().Dot(f.Normal()) < 1e-3:
				
				valid = False
				break
			
		
			newFaces.append(face)
			new_face_verts.add(face.V1)
			new_face_verts.add(face.V2)
			new_face_verts.add(face.V3)

		
		
		if not valid:
			continue



		del vertexFaces[p.A]

		del vertexFaces[p.B]
	


		for f in distinctFaces:
			f.Removed = True
			numFaces -= 1

		for f in newFaces:
	
			numFaces +=1
			try:
				vertexFaces[f.V1].append(f)
			except KeyError:
				vertexFaces[f.V1] = [f]
			
			try:
				vertexFaces[f.V2].append(f)
			except KeyError:
				vertexFaces[f.V2] = [f]


			try:
				vertexFaces[f.V3].append(f)
			except KeyError:
				vertexFaces[f.V3] = [f]


		del vertexPairs[p.A]


		del vertexPairs[p.B]


		seen = set()


		for q in distinctPairs:
		
			q.Removed = True
			try:
				del heap_copy[q]
			except KeyError:
				pass

			a,b = q.A, q.B


			if a == p.A or a == p.B:
				a = v

			if b == p.A or b == p.B:
				b = v

			if b == v:
				a,b = b,a


			if b.Vector in seen:
					continue
			if a not in new_face_verts or b not in new_face_verts:
				continue

			if a == b:
				continue

			seen.add(b.Vector)
			q = Pair(a,b)


			heap_copy[q]=q.Error()

			try:
				vertexPairs[a].append(q)
			except KeyError:
				vertexPairs[a] = [q]
			try:
				vertexPairs[b].append(q)
			except KeyError:
				vertexPairs[b] = [q]

	
	return vertexFaces, vertexPairs, heap_copy, numFaces, target_reached