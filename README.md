#

## Usage

Once the files are in a supported format, the tool can be run after providing the 3D file and the stopping criteria as a fraction. For example if 0.3 is provided, the algorithm will stop when 70% of the faces have been deleted. A list of decreasing fractions, separated by commas, can also be provided. The command below will output three sets of data with 50%, 90%, 99.5% reductions respectively.

    $ python simplify_mesh.py /path/to/file.stl 0.5,0.1,0.005
    


## Details

The original algorithm contracts the edge with the minimal quadric error. For large parts, this can add significant computation time. Other implementations have instead iterated through the edges and contracted any whose error was below a cutoff value, decreasing the time required dramatically .  This version uses a hybrid of the two; first using the cutoff method to decrease the number of overall faces and then if necessary it uses the minimal quadric error. Once faces have been reduced with the cutoff method, finding the minimum error is not as demanding. Additionally, this algorithm preserves boundary edges by adding a weighted matrix to edges with only one neighboring face.

