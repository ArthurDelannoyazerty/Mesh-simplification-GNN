import vtk
import os

def convert_OBJ_to_STL(input_obj_file, output_stl_file, rotate_angle_x=0.0):
    # Crée un objet de type vtkOBJReader pour lire des fichiers OBJ.
    reader = vtk.vtkOBJReader()
    reader.SetFileName(input_obj_file)
    reader.Update()

    # Obtenez les coordonnées des points du modèle.
    polydata = reader.GetOutput()
    points = polydata.GetPoints()

    # Créez une transformation pour effectuer une rotation autour de l'axe X.
    rotation = vtk.vtkTransform()
    rotation.Identity()
    rotation.RotateX(rotate_angle_x)  # Rotation autour de l'axe X.

    # Appliquez la transformation à tous les points du modèle.
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(rotation)
    transformFilter.SetInputData(polydata)
    transformFilter.Update()

    # Obtenez le modèle 3D transformé.
    transformed_polydata = transformFilter.GetOutput()

    # Créez un objet de type vtkSTLWriter pour écrire le modèle 3D final au format STL.
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_stl_file)
    writer.SetInputData(transformed_polydata)
    writer.Write()

if __name__ == "__main__":

    directory_obj = "3d_models\obj"
    directory_stl = "3d_models\stl"
    for filename in os.listdir(directory_obj):
        input_obj_file = os.path.join(directory_obj, filename)
        if os.path.isfile(input_obj_file):
            output_stl_file = os.path.join(directory_stl, filename)[0:-4] + ".stl"
            print(output_stl_file)
    
            # Ajoutez l'angle de rotation autour de l'axe X en degrés ici
            rotate_angle_x = 90.0
            
            convert_OBJ_to_STL(input_obj_file, output_stl_file, rotate_angle_x)
