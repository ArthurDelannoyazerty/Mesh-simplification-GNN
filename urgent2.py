import pdb
import numpy as np
import sys
# Convert to format similar to work
import struct

from triangle import Triangle
#from vector import Vector
from mesh import Mesh
from read_file import *
import os.path
import time
import cProfile

def read_file_into_mesh(file, to_pickle=False):
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
    for i, f in enumerate(faces):
        vec = [0, 0, 0]
        for j in range(3):
            vec[j] = Vector(vertices[f[j]][0], vertices[f[j]][1], vertices[f[j]][2])
        triangles[i] = Triangle(vec[0], vec[1], vec[2])

        amtDone = i / n
        sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))

    return triangles

def get_vertices_faces_from_triangles(triangles):
    vertices = [0] * len(triangles) * 3

    i = 0
    for t in triangles:
        vertices[i * 3] = [t.V1.X, t.V1.Y, t.V1.Z]
        vertices[i * 3 + 1] = [t.V2.X, t.V2.Y, t.V2.Z]
        vertices[i * 3 + 2] = [t.V3.X, t.V3.Y, t.V3.Z]

        i += 1

    faces = np.zeros((len(triangles), 3), dtype=int)
    del_verts = []
    unique_vert_count = 1
    vert_dict = {}
    face_list = []

    for i in range(len(vertices)):
        new_vert = frozenset(vertices[i])
        try:
            face_val = vert_dict[new_vert]
            del_verts.append(i)
        except KeyError:
            face_val = unique_vert_count
            unique_vert_count += 1

        vert_dict[new_vert] = face_val
        faces[i // 3][i % 3] = face_val

    for i in reversed(del_verts):
        del vertices[i]

    vertices = np.array(vertices)
    faces = faces - 1
    return vertices, faces

def write_data_out(vertices, faces, iters, basename):
    print('\nSaving simplified data...')
    np.savetxt(basename + f'_vertices_{iters}.txt', vertices, fmt='%7.4f')
    print('Saved file: ' + basename + f'_vertices_{iters}.txt')
    np.savetxt(basename + f'_faces_{iters}.txt', faces, fmt='%1.1i')
    print('Saved file: ' + basename + f'_faces_{iters}.txt')
def write_stl_file(file_path, vertices, faces):
    # Open the file in binary mode for writing
    with open(file_path, 'wb') as f:
        # Write the STL header
        f.write(b'\0' * 80)

        # Write the number of triangles (faces)
        num_faces = len(faces)
        f.write(num_faces.to_bytes(4, byteorder='little'))

        # Write each triangle (face) to the file
        for face in faces:
            # Write the normal vector (not used in your code, set to zero)
            f.write(struct.pack('<3f', 0.0, 0.0, 0.0))

            # Write the three vertices of the triangle
            for vertex_index in face:
                vertex = vertices[vertex_index]
                f.write(struct.pack('<3f', vertex[0], vertex[1], vertex[2]))

            # Write attribute (not used, set to zero)
            f.write(b'\0' * 2)
def write_stl_out(vertices, faces, basename, iters):
    print('\nSaving simplified STL...')
    write_stl_file(basename + f'_simplified_{iters}.stl', vertices, faces)
    print('Saved file: ' + basename + f'_simplified_{iters}.stl')

if __name__ == '__main__':
    def main():
        start_time = time.time()

        # Check if the correct number of command line arguments is provided
        if len(sys.argv) != 3:
            print("Usage: python script.py input.stl simplification_factor")
            sys.exit(1)

        file = sys.argv[1]
        fraction = float(sys.argv[2])

        basename = os.path.splitext(os.path.basename(file))[0]
        part_file_extension = os.path.splitext(file)[1]

        if part_file_extension == '.stl':
            vertices, faces = read_stl_file(file)
        elif part_file_extension == '.obj':
            vertices, faces = read_obj_file(file)

        triangles = create_triangles(vertices, faces)

        # Save original version
        ver, faces = get_vertices_faces_from_triangles(triangles)
        original_num_faces = len(faces)
        write_data_out(ver, faces, 'original', basename)
        del ver
        del faces

        mesh = Mesh(triangles)
        triangles = mesh.simplify(fraction, original_num_faces)

        ver, faces = get_vertices_faces_from_triangles(triangles)
        write_data_out(ver, faces, f'{fraction:.2f}', basename)
        target_num_faces = len(faces)
        write_stl_out(ver, faces, basename, f'{fraction:.2f}')
        difference = original_num_faces - target_num_faces

        end_time = time.time()
        print(f'Difference between Original and Target Number of Faces: {difference}')
        print(f'\nNs/Norg: {fraction} ==> Finished in: {end_time - start_time} seconds')

    cProfile.run('main()')
