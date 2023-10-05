import vtk

def convert_OBJ_to_STL(input_obj_file, output_stl_file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(input_obj_file)
    reader.Update()

    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(reader.GetOutputPort())
    surface = triangle.GetOutputPort()

    normals = vtk.vtkPolyDataNormals()
    normals.SetAutoOrientNormals(False)
    normals.SetFlipNormals(True)
    normals.SetSplitting(False)
    normals.SetFeatureAngle(90.0)
    normals.ConsistencyOn()
    normals.SetInputConnection(surface)
    surface = normals.GetOutputPort()

    fillHoles = vtk.vtkFillHolesFilter()
    fillHoles.SetHoleSize(1000.0)
    fillHoles.SetInputConnection(surface)
    surface = fillHoles.GetOutputPort()

    writer = vtk.vtkSTLWriter()
    writer.SetFileTypeToBinary()
    writer.SetInputConnection(surface)
    writer.SetFileName(output_stl_file)
    writer.Update()
    writer.Write()


if __name__ == "__main__":
    input_obj_file = 'C:\\Users\\dvora\\OneDrive\\Documents\\Mesh-simplification-GNN-mesh\\Models\\bunny.obj'
    output_stl_file = 'C:\\Users\\dvora\\OneDrive\\Documents\\Mesh-simplification-GNN-mesh\\Models\\bunny.stl'
    convert_OBJ_to_STL(input_obj_file, output_stl_file)
