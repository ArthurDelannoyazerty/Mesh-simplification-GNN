import trimesh
import os
import numpy as np


def convert_OBJ_to_STL(input_obj_file, output_stl_file):
    mesh = trimesh.load(input_obj_file)
    angle = np.radians(90)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
    mesh.apply_transform(rotation_matrix)
    mesh.export(output_stl_file, file_type='stl')


if __name__ == "__main__":
    directory_obj = "3d_models\obj"
    directory_stl = "3d_models\stl"
    for filename in os.listdir(directory_obj):
        input_obj_file = os.path.join(directory_obj, filename)
        if os.path.isfile(input_obj_file):
            output_stl_file = os.path.join(directory_stl, filename)[0:-4] + ".stl"
            convert_OBJ_to_STL(input_obj_file, output_stl_file)
