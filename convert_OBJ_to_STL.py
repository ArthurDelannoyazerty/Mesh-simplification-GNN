import vtk
import os
import aspose.threed as a3d


def convert_OBJ_to_STL(input_obj_file, output_stl_file):
    scene = a3d.Scene.from_file(input_obj_file)
    scene.save(output_stl_file)

if __name__ == "__main__":

    directory_obj = "3d_models\obj"
    directory_stl = "3d_models\stl"
    for filename in os.listdir(directory_obj):
        input_obj_file = os.path.join(directory_obj, filename)
        if os.path.isfile(input_obj_file):
            output_stl_file = os.path.join(directory_stl, filename)[0:-4] + ".stl"
            print(output_stl_file)

            convert_OBJ_to_STL(input_obj_file, output_stl_file)
