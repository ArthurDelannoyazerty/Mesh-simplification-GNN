import pdb
import numpy as np
import sys
#convert to format similar to work

from triangle import Triangle
#from vector import Vector
from mesh import Mesh
from read_file import *
import os.path
import time

def read_file_into_mesh(file, to_pickle = False):
	vertex_num = 0
	vertices = {}
	faces = []
	vert_key = []
	num_lines = len(open(file).readlines())
	line_num = 0.0

	print('Converting to work format...\n')
	triangles = []
	vectors = []
	for line in open(file):
		line_num += 1.0
		split_line = line.split()
	
		if split_line[0] == 'vertex':
			
		
			vectors.append(Vector(float(split_line[1]), float(split_line[2]), float(split_line[3])))


		elif split_line[0] == 'endloop':
			triangles.append(Triangle(vectors[0], vectors[1], vectors[2]))
			vectors = []

		amtDone = line_num / num_lines
		sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))




	return triangles

def create_triangles(vertices, faces):
	print('\nCreating triangle objects...')
	n = float(len(faces))

	triangles = [0] * len(faces)
	for i,f in enumerate(faces):
		vec = [0,0,0]
		for j in range(3):
			# import pdb
			# pdb.set_trace()
			vec[j] = Vector(vertices[f[j]][0], vertices[f[j]][1],vertices[f[j]][2])
		triangles[i] = Triangle(vec[0],vec[1], vec[2])

		amtDone = i / n
		sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))

	return triangles


def get_vertices_faces_from_triangles(triangles):
	vertices = [0] * len(triangles)*3


	i = 0
	for t in triangles:
		vertices[i*3] = [t.V1.X, t.V1.Y, t.V1.Z]
		vertices[i*3 + 1] = [t.V2.X, t.V2.Y, t.V2.Z]
		vertices[i*3 + 2] = [t.V3.X, t.V3.Y, t.V3.Z]

		i+=1
	
	faces = np.zeros((len(triangles),3),dtype = int)
	del_verts = []
	unique_vert_count =1
	vert_dict = {}
	face_list = []

	for i in range(len(vertices)):
		new_vert = frozenset(vertices[i])
		try:
			face_val = vert_dict[new_vert]
			del_verts.append(i)
		except KeyError:
			face_val = unique_vert_count
			unique_vert_count +=1
		
		vert_dict[new_vert] = face_val
		faces[i//3][i%3] = face_val

	for i in reversed(del_verts):
		del vertices[i]

	vertices = np.array(vertices)
	faces = faces - 1
	return vertices, faces

def write_data_out(vertices, faces,iters, basename):
	print('\nSaving simplified data...')
	np.savetxt(basename + '_vertices_'+str(iters)+'.txt',vertices, fmt = '%7.4f')
	print('Saved file: ' + basename + '_vertices_'+str(iters)+'.txt')
	np.savetxt(basename + '_faces_'+str(iters)+'.txt',faces, fmt = '%1.1i')
	print('Saved file: ' + basename + '_faces_'+str(iters)+'.txt')
	

if __name__ == '__main__':
	start_time = time.time()
	inputs = sys.argv

	file = sys.argv[1]

	fraction = sys.argv[2]
	if ',' in fraction:
		fraction = fraction.split(',')
		fraction = list(map(float,fraction))
	else:
		fraction = [float(fraction)]

	#file = "/home/adam/3d_facets/stormtrooper/helmet-ascii.stl"
	#file = "/home/adam/3d_facets/cheval.stl"
	basename = os.path.splitext(os.path.basename(file))[0]
	part_file_extension = os.path.splitext(file)[1]

	if part_file_extension == '.stl':
		vertices, faces = read_stl_file(file)
	elif part_file_extension == '.obj':
		vertices, faces = read_obj_file(file)


	triangles = create_triangles(vertices, faces)

	# Save original version
	ver, faces = get_vertices_faces_from_triangles(triangles)
	write_data_out(ver,faces,'original', basename)
	del ver
	del faces

	if len(fraction) == 1:
		quit()

	original_num_faces = len(triangles)
	for frac in fraction:

		mesh = Mesh(triangles)
		triangles = mesh.simplify(frac,original_num_faces)

		ver, faces = get_vertices_faces_from_triangles(triangles)

		write_data_out(ver,faces,frac, basename)
	end_time = time.time()
	print('\nFinished in: ' + str(end_time - start_time) + ' seconds')