import numpy as np
import sys
import os.path
import time
from triangle import Triangle
from mesh import Mesh
from read_file import read_stl_file, read_obj_file
from vector import Vector

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

def write_data_out(vertices, faces,iters, basename):
    print('\nSaving simplified data...')
    np.savetxt(basename + '_vertices_'+str(iters)+'.txt',vertices, fmt = '%7.4f')
    print('Saved file: ' + basename + '_vertices_'+str(iters)+'.txt')
    np.savetxt(basename + '_faces_'+str(iters)+'.txt',faces, fmt = '%1.1i')
    print('Saved file: ' + basename + '_faces_'+str(iters)+'.txt')

def get_vertices_faces_from_triangles(triangles):
    vertices = [0] * len(triangles)*3

    i = 0
    for t in triangles:
        vertices[i*3] = [t.V1.X, t.V1.Y, t.V1.Z]
        vertices[i*3 + 1] = [t.V2.X, t.V2.Y, t.V2.Z]
        vertices[i*3 + 2] = [t.V3.X, t.V3.Y, t.V3.Z]

        i += 1

    faces = np.zeros((len(triangles),3), dtype=int)
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
        faces[i//3][i%3] = face_val

    for i in reversed(del_verts):
        del vertices[i]

    vertices = np.array(vertices)
    faces = faces - 1
    return vertices, faces

def laplacian_simplify(mesh, iterations=1):
    for _ in range(iterations):
        mesh.laplacian_smooth()

if __name__ == '__main__':
    start_time = time.time()
    inputs = sys.argv

    file = sys.argv[1]

    fraction = sys.argv[2]
    if ',' in fraction:
        fraction = fraction.split(',')
        fraction = list(map(float, fraction))
    else:
        fraction = [float(fraction)]

    basename = os.path.splitext(os.path.basename(file))[0]
    part_file_extension = os.path.splitext(file)[1]

    if part_file_extension == '.stl':
        vertices, faces = read_stl_file(file)
    elif part_file_extension == '.obj':
        vertices, faces = read_obj_file(file)

    triangles = create_triangles(vertices, faces)
    
    # Save original version
    ver, faces = get_vertices_faces_from_triangles(triangles)
    write_data_out(ver, faces, 'original', basename)
    del ver
    del faces

    if len(fraction) == 1:
        quit()

    original_num_faces = len(triangles)
    for frac in fraction:
        mesh = Mesh(triangles)
        laplacian_simplify(mesh, iterations=5)  # You can adjust the number of iterations
        triangles = mesh.get_triangles()

        ver, faces = get_vertices_faces_from_triangles(triangles)
        write_data_out(ver, faces, frac, basename)

    end_time = time.time()
    print('\nFinished in: Laplacian ' + str(end_time - start_time) + ' seconds')

class Mesh:
    def __init__(self, triangles):
        self.triangles = triangles

    def laplacian_smooth(self):
        vertex_neighbors = self.find_vertex_neighbors()

        # Perform Laplacian smoothing
        for i in range(len(self.triangles)):
            avg_vertex = np.mean([t.V1, t.V2, t.V3], axis=0)
            self.triangles[i].V1 = avg_vertex



    def get_triangles(self):
        return self.triangles
