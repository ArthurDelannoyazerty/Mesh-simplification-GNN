import numpy as np
import sys
from .vector import Vector
from .triangle import Triangle
import struct
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

def read_stl(file_path):
    vertices = []
    faces = []

    with open(file_path, 'rb') as f:
        # Read the header (80 bytes), which can be ignored
        f.read(80)

        # Read the number of triangles (faces) as an unsigned 32-bit integer
        num_faces_bytes = f.read(4)
        num_faces = struct.unpack('<I', num_faces_bytes)[0]

        for _ in range(num_faces):
            # Read the normal vector (which can be ignored)
            f.read(12)

            # Read the three vertices of the triangle as 32-bit floating-point numbers
            vertices.append(struct.unpack('<3f', f.read(12)))
            vertices.append(struct.unpack('<3f', f.read(12)))
            vertices.append(struct.unpack('<3f', f.read(12)))

            # Read the attribute (which can be ignored)
            f.read(2)

            # Add the face indices
            faces.append([len(faces) * 3, len(faces) * 3 + 1, len(faces) * 3 + 2])

    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)

    return vertices, faces


