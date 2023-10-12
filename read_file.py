
import numpy as np
import sys
from vector import Vector
from triangle import Triangle

def read_obj_file(file):
	

	vertices = []
	faces = []
	num_lines = len(open(file).readlines())
	line_num = 0.0
	print('Reading .obj file...')
	for line in open(file, 'r'):
		line_num += 1.0
		split_line = line.split()

		if len(split_line) == 0:
			continue

		if split_line[0] == 'v':
			vertices.append([split_line[1], split_line[2], split_line[3]])
			continue

		if split_line[0] == 'f':
			
			new_face = [0,0,0]
			for i in range(3):
				f = split_line[i+1]
				slash_ind = f.find('/')
				if slash_ind == -1:
					new_face[i] = f
				else:
					new_face[i] = f[:slash_ind]
			faces.append(new_face)

		amtDone = line_num / num_lines
		sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))
	
	faces = np.array(faces, dtype = int) - 1
	vertices = np.array(vertices,dtype = float)
	return vertices, faces


def read_stl_file(file):

	vertices = []
	faces = []
	num_lines = len(open(file).readlines())
	line_num = 0.0
	face_num = 0
	print('Reading .stl file...\n')

	for line in open(file):
		line_num += 1.0
		split_line = line.split()
	
		if split_line[0] == 'vertex':
			
		
			vertices.append([float(split_line[1]), float(split_line[2]), float(split_line[3])])


		elif split_line[0] == 'endloop':
			faces.append([face_num*3, face_num*3+1, face_num*3+2])
			face_num +=1
		amtDone = line_num / num_lines
		sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))


	faces = np.array(faces, dtype = int)
	vertices = np.array(vertices,dtype = float)

	return vertices, faces 




