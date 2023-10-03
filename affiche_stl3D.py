import vtk

def main():
    # Create a reader
    reader = vtk.vtkSTLReader()
    reader.SetFileName("Models/alien_pilot_standing.stl")

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # A renderer and render window
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # An interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actors to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1) # Background color

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__ == "__main__":
    main()