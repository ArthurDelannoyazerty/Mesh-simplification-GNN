import vtk

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
    input_obj_file = 'bunny.obj'
    output_stl_file = 'bunny_rootated.stl'
    
    # Ajoutez l'angle de rotation autour de l'axe X en degrés ici
    rotate_angle_x = 90.0
    
    convert_OBJ_to_STL(input_obj_file, output_stl_file, rotate_angle_x)
